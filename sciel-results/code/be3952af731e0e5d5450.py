import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit, use 120s as safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p = 100
    # This is the trivial exact decomposition.
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k)
    
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Check if we're still within time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # This shouldn't happen for such a small problem, but guard just in case
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
