import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    timeout = 140
    
    n, m, p = DIMENSIONS
    
    # Use Schoolbook decomposition (Rank n*m*p) for robustness
    # This guarantees an exact decomposition of the dense matrix multiplication tensor.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j, col] = 1
                # V index: j * p + k
                V[j * p + k, col] = 1
                # W index: k * n + i (as per contract)
                W[k * n + i, col] = 1
                col += 1
    
    if time.time() - start_time > timeout - 5:
        pass  # Return immediately
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
