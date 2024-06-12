import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

import matplotlib.pyplot as plt


train=pd.read_csv(r"E:\Machine Learning Projects\Titanic Survival\Data\titanic\train.csv")
test=pd.read_csv(r"E:\Machine Learning Projects\Titanic Survival\Data\titanic\test.csv")




impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)


train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

predictors = ['Pclass', 'IsFemale', 'Age']
x_train=train[predictors].values
x_test=test[predictors].values
y_train=train['Survived'].values

model=LogisticRegressionCV()
model.fit(x_train,y_train)
predict=model.predict(x_test)
print(predict)


from sklearn.model_selection import cross_val_score
model = LogisticRegression(C=8)
scores = cross_val_score(model, x_train, y_train, cv=4)
print(scores)






