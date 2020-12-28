# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:25:22 2020

@author: shara
"""

import pandas as pd
import numpy as np
wine = pd.read_csv("G:\DS Assignments\PCA\wine.csv")
wine.head()
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
wine1 = wine.iloc[: , 1:]


wine_scale = scale(wine1)
pca = PCA(n_components=3)
pca_values = pca.fit_transform(wine_scale)
var = pca.explained_variance_ratio_
pca.components_[1]
pca_values.shape
var1 = np.cumsum(np.round(var, decimals = 4)*100)
import matplotlib.pylab as plt
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')

plt.scatter(PCA_comp[0], PCA_comp[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


PCA_comp = pd.DataFrame(pca_values)

ks = range(1, 20)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(PCA_comp.iloc[:,:3])
    
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


model = KMeans(n_clusters = 3)
model.fit(PCA_comp)
model.labels_
md = pd.Series(model.labels_)
wine["cluster"] = md
wine = wine.iloc[ : , [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(PCA_comp, method = "complete" , metric = "euclidean")
plt.figure(figsize = (15,5)) ; plt.title("H_Clustering_Dendrogram");plt.xlabel("index");plt.ylabel("Distance")
sch.dendrogram(z)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete',affinity = "euclidean").fit(PCA_comp)
cluster_labels  = pd.Series(h_complete.labels_)
wine["Cluster_labels_H"] = cluster_labels
wine = wine.iloc[ : , [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15]]
