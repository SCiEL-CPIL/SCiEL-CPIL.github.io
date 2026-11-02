import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is identity of size n*m
    for r in range(rank):
        U[r][r] = 1
        
    # V: shape (m*p, rank)
    # For column r, i = r // m, j = r % m
    # V[row, r] = 1 if row corresponds to (j, k) for the specific j from r, for all k
    for r in range(rank):
        i_val = r // m
        j_val = r % m
        for k in range(p):
            v_row = j_val * p + k
            V[v_row][r] = 1
            
    # W: shape (p*n, rank)
    # For column r, i = r // m, j = r % m
    # W[row, r] = 1 if row corresponds to (k, i) for the specific i from r, for all k
    for r in range(rank):
        i_val = r // m
        j_val = r % m
        for k in range(p):
            w_row = k * n + i_val
            W[w_row][r] = 1
            
    return {"U": U, "V": V, "W": W}
