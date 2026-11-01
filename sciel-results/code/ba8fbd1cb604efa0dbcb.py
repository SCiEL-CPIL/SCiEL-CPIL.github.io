import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Standard matrix multiplication algorithm
    # Rank = n * m * p
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Iterate over all indices of A, B, C
    # A is n x m
    # B is m x p
    # C is n x p (stored as p x n in W with index k*n + i)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in A: i*m + j
                idx_a = i * m + j
                
                # Index in B: j*p + k
                idx_b = j * p + k
                
                # Index in W (C): k*n + i
                # Convention: W[k*n + i, r]
                idx_c = k * n + i
                
                U[idx_a][r] = 1
                V[idx_b][r] = 1
                W[idx_c][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
