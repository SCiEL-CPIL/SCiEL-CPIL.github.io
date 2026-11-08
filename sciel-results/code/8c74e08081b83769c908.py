import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: i=0
    for r in range(4):
        j = r
        # U
        u_idx = 0 * m + j
        U[u_idx][r] = 1
        
        # V
        for k in range(p):
            v_idx = j * p + k
            V[v_idx][r] = 1
            
        # W
        for k in range(p):
            w_idx = k * n + 0
            W[w_idx][r] = 1
            
    # Block 2: i=1
    for r in range(4, 8):
        j = r - 4
        # U
        u_idx = 1 * m + j
        U[u_idx][r] = 1
        
        # V
        for k in range(p):
            v_idx = j * p + k
            V[v_idx][r] = 1
            
        # W
        for k in range(p):
            w_idx = k * n + 1
            W[w_idx][r] = 1

    return {"U": U, "V": V, "W": W}
