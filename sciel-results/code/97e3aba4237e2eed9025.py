"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Direct sum of 6 copies of 2x2x2 multiplication (rank 4 each) = rank 24
    n, m, p = DIMENSIONS
    # Partition 12 columns into 6 groups of 2
    num_groups = p // m  # 12 // 2 = 6
    rank_per_block = n * m * m  # 4 for 2x2x2
    total_rank = num_groups * rank_per_block  # 24
    
    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 24
    V = [[0] * total_rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x 24
    
    col = 0
    for g in range(num_groups):
        for i in range(n):
            for j in range(m):
                # For block g, columns k = g*m + k_local where k_local in 0..m-1
                for k_local in range(m):
                    k = g * m + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
