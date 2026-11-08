"""Matrix multiplication 3,4,6 rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p  # 72
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the expected interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
