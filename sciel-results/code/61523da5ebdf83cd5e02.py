import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank: 2*3*4 = 24 (trivial decomposition)
    rank = 24
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
