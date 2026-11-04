"""Matrix multiplication 2,3,6 rank decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    total = n * m * p  # 36
    
    # Schoolbook rank 36 decomposition
    U = np.zeros((n * m, total), dtype=float)
    V = np.zeros((m * p, total), dtype=float)
    W = np.zeros((p * n, total), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
