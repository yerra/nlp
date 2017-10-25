# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def calc_performance(cm):
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    total = tn + fp + fn + tp
    accuracy = round((tn + tp)*100/float(total),2)
    precision = round((tp*100)/float(fp + tp),2)
    recall  = round((tp*100)/(fn + tp),2)
    f1Score  = round(2*precision*recall/float(precision+recall),2)    
    return (tp,tn,fp,fn,accuracy,precision,recall,f1Score)


#setting working directory

os.chdir('D:\Dataset')

# Importing the Data sets
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter ='\t',quoting = 3)

# Cleaning the text
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,len(dataset['Review'])):
    ## Step1: Keep non character 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # Step2: Converting to Lowercase
    review = review.lower()
    
    #Step3: Remove stopwords
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #Step4: Stemming
    ps = PorterStemmer()
    review =[ps.stem(word) for word in review]
    
    #Step5:  Join the word to make sentence
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the Bag of Words Model
from sklearn.feature_extraction.text import  CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Dimensionality Reduction 
#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 510)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

## Model1 : Logistic Regression Classification
if 1:
# Fitting Logistics Regression  to the  Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train,y_train)
    
    #Predict Test results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    log_regression_cm = confusion_matrix(y_test, y_pred)
    lr_performance = calc_performance(log_regression_cm)
    
##Model2: KNN 
if 1:
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
                                
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    knn_cm = confusion_matrix(y_test, y_pred)
    knn_performance = calc_performance(knn_cm)

# Model3: SVM
if 1:
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)  
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    svm_cm = confusion_matrix(y_test, y_pred)
    svm_performance = calc_performance(svm_cm)


## Mode4 : Naive Bayes
if 1:
    
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    naive_bayes_cm = confusion_matrix(y_test, y_pred)
    nb_performance = calc_performance(naive_bayes_cm)
    
## Model5:Decision Tree
if 1:
# Fitting Classifier to the  Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifier.fit(X_train,y_train)    
    
    #Predict Test results
    y_pred = classifier.predict(X_test)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    decision_tree_cm = confusion_matrix(y_test,y_pred)
    dt_performance = calc_performance(decision_tree_cm)

if 1:
## Model6: Random Forest
# Fitting Classifier to the  Training set
# Create  your classifier  here
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train,y_train)
    #Predict Test results
    y_pred = classifier.predict(X_test)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    random_forest_cm = confusion_matrix(y_test,y_pred)
    rf_performance = calc_performance(random_forest_cm)
    
## Model Performance 
data = data = [lr_performance,knn_performance,svm_performance,nb_performance,dt_performance,rf_performance]
index = ['log_reg','knn','svm','naive bayes','Decision Tree','random forest']
name = ['TP','TN','FP','FN','acc','prec','rec','f1']
df = pd.DataFrame(data,columns=name,index=index)

    





