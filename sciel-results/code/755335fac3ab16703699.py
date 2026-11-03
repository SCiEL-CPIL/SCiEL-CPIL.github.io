"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # The minimal rank for the matrix multiplication tensor 
    # (defined by T[i, j, j', k] = delta_{j, j'}) is m.
    # This is because the slice for any fixed i, k is the identity matrix I_m,
    # which has rank m.
    rank = m
    
    U = [[0] * rank for _ in range(n_m)]
    V = [[0] * rank for _ in range(m_p)]
    W = [[0] * rank for _ in range(p_n)]
    
    for r in range(rank):
        # U[i*m + j, r] = 1 if j == r else 0
        for i in range(n):
            for j in range(m):
                if j == r:
                    U[i * m + j][r] = 1
        
        # V[j'*p + k, r] = 1 if j' == r else 0
        for j_prime in range(m):
            for k in range(p):
                if j_prime == r:
                    V[j_prime * p + k][r] = 1
        
        # W[k*n + i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
