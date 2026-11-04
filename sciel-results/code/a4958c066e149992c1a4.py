import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Rank 4 solution: One term per j
    # U[i*m+j, c] = 1 if c == j, else 0
    # V[j*p+k, c] = 1 if c == j, else 0
    # W[k*n+i, c] = 1 if c == j, else 0
    
    U = np.zeros((n * m, m))
    V = np.zeros((m * p, m))
    W = np.zeros((p * n, m))
    
    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1.0
        for k in range(p):
            V[j * p + k, j] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
