import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 8 decomposition
    # T[i, j, k] = 1 for all i, j, k
    # We use r = i * m + j
    rank = n * m
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U[i, j, r] = 1
            U[i * m + j, r] = 1.0
            
            # V[j, k, r] = 1 for all k
            for k in range(p):
                V[j * p + k, r] = 1.0
            
            # W[k, i, r] = 1 for all k
            for k in range(p):
                W[k * n + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
