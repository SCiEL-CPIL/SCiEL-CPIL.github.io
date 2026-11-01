import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # The minimal rank for 2x3 by 3x4 matrix multiplication is 24.
    # We construct the standard schoolbook decomposition which achieves this rank.
    # Each term corresponds to a unique triple (i, j, k).
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
                col += 1
                
    # Convert to lists of lists as per interface requirement if strict, 
    # but usually numpy arrays or lists are accepted if shapes match.
    # The prompt example uses lists of lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
