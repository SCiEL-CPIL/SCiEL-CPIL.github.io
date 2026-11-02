import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Check deadline
    if time.time() > deadline:
        # Return what we have, or a valid fallback
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
