import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    start_time = time.time()
    timeout = 110.0
    
    # Schoolbook solution
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
