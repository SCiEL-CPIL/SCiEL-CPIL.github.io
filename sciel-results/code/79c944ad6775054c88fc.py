import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Construct the schoolbook decomposition
    # Each term corresponds to a specific (i, j, k) multiplication
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                U[i * m + j, col] = 1.0
                # V index: row j*p + k
                V[j * p + k, col] = 1.0
                # W index: row k*n + i (following the official trace convention W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists as per interface requirement
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
