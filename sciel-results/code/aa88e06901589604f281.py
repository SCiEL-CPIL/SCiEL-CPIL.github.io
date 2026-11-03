import time
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    R = m * p  # 12
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = [[0.0 for _ in range(R)] for _ in range(dim_u)]
    V = [[0.0 for _ in range(R)] for _ in range(dim_v)]
    W = [[0.0 for _ in range(R)] for _ in range(dim_w)]
    
    # V is Identity
    for r in range(R):
        V[r][r] = 1.0
        
    # W is All Ones
    for i in range(dim_w):
        for r in range(R):
            W[i][r] = 1.0
            
    # U construction
    # U[i*m + j, r] = 1 if r is in the block for j
    # Block for j is r in {4*j, 4*j+1, 4*j+2, 4*j+3}
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                r_idx = j * p + k
                U[u_idx][r_idx] = 1.0
                
    return {"U": U, "V": V, "W": W}
