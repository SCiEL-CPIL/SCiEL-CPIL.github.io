import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 100  # safety margin
    
    # Try rank 1 first
    rank = 1
    U = np.ones((n * m, rank), dtype=object)
    V = np.ones((m * p, rank), dtype=object)
    W = np.ones((p * n, rank), dtype=object)
    
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: trivial rank n*m*p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
