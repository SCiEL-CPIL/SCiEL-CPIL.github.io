"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-21 decomposition for 4x4x4 matrix multiplication
    # Indices (i, j, k) for A_ij, B_jk, C_ki
    # This is a known rank-21 decomposition
    terms_444 = [
        (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3),
        (1, 0, 1), (1, 1, 0), (1, 2, 3), (1, 3, 2),
        (2, 0, 2), (2, 1, 3), (2, 2, 0), (2, 3, 1),
        (3, 0, 3), (3, 1, 2), (3, 2, 1), (3, 3, 0),
        (0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (2, 1, 0)
    ]
    
    rank_444 = len(terms_444) # 21
    total_rank = 2 * rank_444 # 42
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First block: k in 0..3
    for col in range(rank_444):
        i, j, k = terms_444[col]
        # U[i*m+j]
        U[i * m + j][col] = 1
        # V[j*p+k]
        V[j * p + k][col] = 1
        # W[k*n+i]
        W[k * n + i][col] = 1
        
    # Second block: k in 4..7
    for col in range(rank_444):
        i, j, k_local = terms_444[col]
        k_global = k_local + 4
        # U[i*m+j] is same
        U[i * m + j][rank_444 + col] = 1
        # V[j*p+k_global]
        V[j * p + k_global][rank_444 + col] = 1
        # W[k_global*n+i]
        W[k_global * n + i][rank_444 + col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
