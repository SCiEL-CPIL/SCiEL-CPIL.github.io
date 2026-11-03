import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    rank = n * m * p  # Schoolbook rank 216
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U[i, k, r] = 1
                U[i * m + k, r_idx] = 1.0
                # V[k, j, r] = 1
                V[k * p + j, r_idx] = 1.0
                # W[j, i, r] = 1
                W[j * n + i, r_idx] = 1.0
                r_idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
