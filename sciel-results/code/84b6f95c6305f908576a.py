import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        k = r % p
        temp = r // p
        i = temp % n
        j = temp // n
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
