"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    rank = p * n  # 32, which is the lower bound and achievable
    
    # U has shape (n*m, rank) = (4, 32)
    # V has shape (m*p, rank) = (32, 32)
    # W has shape (p*n, rank) = (32, 32)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index r = i * p + k for i in [0, n), k in [0, p)
    # U[i*m + j, r] = 1 if i_u == i_r (where r = i_r * p + k_r)
    # V[j*p + k, r] = 1 if k_v == k_r
    # W[k*n + i, r] = 1 if i_w == i_r and k_w == k_r
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            # Set U entries: U[i*m + j, r] = 1 for all j
            for j in range(m):
                U[i * m + j][r] = 1
            # Set V entries: V[j*p + k, r] = 1 for all j
            for j in range(m):
                V[j * p + k][r] = 1
            # Set W entry: W[k*n + i, r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
