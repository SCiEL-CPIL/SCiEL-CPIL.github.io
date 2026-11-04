import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Check time
    if time.time() > deadline:
        # Should not happen for schoolbook, but just in case
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
