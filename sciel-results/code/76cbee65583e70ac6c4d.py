"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        i_c = c // m
        s_c = c % m
        
        row_u = i_c * m + s_c
        U[row_u][c] = 1
        
        for k in range(p):
            row_v = s_c * p + k
            V[row_v][c] = 1
            
        for k in range(p):
            row_w = k * n + i_c
            W[row_w][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
