"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # Optimized decomposition using rank 1 updates per column j
    # Rank = m = 5
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each j in 0..m-1, we use one term alpha = j
    for j in range(m):
        alpha = j
        
        # U: Index i*m + j
        # U[i*m + j, alpha] = 1 for all i
        for i in range(n):
            idx_u = i * m + j
            U[idx_u][alpha] = 1
            
        # V: Index j*p + k
        # V[j*p + k, alpha] = 1 for all k
        for k in range(p):
            idx_v = j * p + k
            V[idx_v][alpha] = 1
            
        # W: Index k*n + i
        # W[k*n + i, alpha] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                W[idx_w][alpha] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
