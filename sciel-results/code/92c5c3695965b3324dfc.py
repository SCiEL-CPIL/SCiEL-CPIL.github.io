import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    R = n * m * p  # 45
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Schoolbook decomposition: one term per (i, j, k) triple
    for j in range(m):
        for i in range(n):
            for k in range(p):
                r = j * (n * p) + i * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
    # Quick verification that this is valid
    # The decomposition should satisfy: for each (i,k), sum_r U[i*m + r_u, r] * V[r_v, r] * W[k*n + i, r] = A[i,:] @ B[:,k]
    # This is exact by construction.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
