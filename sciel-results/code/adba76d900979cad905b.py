import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
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
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
