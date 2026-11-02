import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 5, 5
    
    # Trivial decomposition: Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i * m + j
                u_idx = i * m + j
                # V index: (j, k) -> j * p + k
                v_idx = j * p + k
                # W index: (k, i) -> k * n + i
                w_idx = k * n + i
                
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                
                idx += 1
                
                # Time check
                if time.time() > deadline - 5:
                    # Return whatever we have (which should be complete for trivial)
                    pass

    return {'U': U, 'V': V, 'W': W}
