import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return what we have if timeout (unlikely for this size)
                    break
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early, we have an invalid decomposition.
    # But 96 iterations is < 1ms.
    
    return {"U": U, "V": V, "W": W}
