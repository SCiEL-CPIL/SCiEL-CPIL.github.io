import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
