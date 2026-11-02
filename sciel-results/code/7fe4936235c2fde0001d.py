import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # The schoolbook rank is n*m*p
    R = n * m * p
    
    U = np.zeros((num_u, R), dtype=float)
    V = np.zeros((num_v, R), dtype=float)
    W = np.zeros((num_w, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
