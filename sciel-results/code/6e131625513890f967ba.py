import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize arrays
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Check time
    if time.time() - start_time > deadline:
        # This should never happen for this small problem
        pass
        
    return {"U": U, "V": V, "W": W}
