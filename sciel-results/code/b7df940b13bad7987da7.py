import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early, the decomposition might be incomplete/invalid.
    # But the loop is very fast, so this shouldn't happen.
    # To be safe, if we didn't finish, we should probably return a known valid one?
    # But we can't construct a valid one partially.
    # So we assume it finishes.
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
