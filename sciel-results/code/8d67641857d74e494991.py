"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Partition of (i, k) pairs for each j
    # i in 0..2, k in 0..4
    # 5 groups of 3, each with distinct i and distinct k
    groups = [
        [(0, 0), (1, 1), (2, 2)],
        [(0, 1), (1, 2), (2, 3)],
        [(0, 2), (1, 3), (2, 4)],
        [(0, 3), (1, 4), (2, 0)],
        [(0, 4), (1, 0), (2, 1)]
    ]
    
    for j in range(m):
        for g_idx, group in enumerate(groups):
            col = j * 5 + g_idx
            for i, k in group:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
