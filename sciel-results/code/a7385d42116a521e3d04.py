import numpy as np
from scipy.spatial import cKDTree

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # K nearest neighbors
    K = 8
    
    # Build KDTree for efficient k-NN query
    # We don't have the points, only the distance matrix.
    # We can find k-NN from the distance matrix directly.
    
    # For each i, find indices of K smallest distances (excluding self)
    # distance_matrix[i, i] is 0.
    
    eta = np.full((n, n), 1e-9)
    
    for i in range(n):
        # Get distances from i to all others
        dists = distance_matrix[i].copy()
        dists[i] = np.inf # Exclude self
        
        # Find K smallest indices
        # np.argpartition is faster than sort for this
        k_indices = np.argpartition(dists, K)[:K]
        
        for j in k_indices:
            # Set heuristic based on inverse distance
            # Using inverse distance squared might be sharper
            eta[i, j] = 1.0 / (distance_matrix[i, j] ** 2 + 1e-9)
            eta[j, i] = 1.0 / (distance_matrix[i, j] ** 2 + 1e-9)

    return eta
