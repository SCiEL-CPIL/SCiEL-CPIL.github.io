"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Use rank = n * m strategy
    # r encodes (i, j)
    rank = n * m
    
    # U: shape (n*m, rank)
    # U[i*m + j, r] = 1 if r == i*m + j
    U = [[0] * rank for _ in range(n * m)]
    for r in range(rank):
        i = r // m
        j = r % m
        U[i * m + j][r] = 1
        
    # V: shape (m*p, rank)
    # V[j*p + k, r] = 1 if j == r % m
    V = [[0] * rank for _ in range(m * p)]
    for r in range(rank):
        j_r = r % m
        for k in range(p):
            V[j_r * p + k][r] = 1
            
    # W: shape (p*n, rank)
    # W[k*n + i, r] = 1 if i == r // m
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i_r = r // m
        for k in range(p):
            W[k * n + i_r][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
