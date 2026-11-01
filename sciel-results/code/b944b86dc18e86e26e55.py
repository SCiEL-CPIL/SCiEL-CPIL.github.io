import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        # Decompose c into i, j, k
        # c = i * (m * p) + j * p + k
        k = c % p
        c2 = c // p
        j = c2 % m
        i = c2 // m
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][c] = 1
        V[v_idx][c] = 1
        W[w_idx][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
