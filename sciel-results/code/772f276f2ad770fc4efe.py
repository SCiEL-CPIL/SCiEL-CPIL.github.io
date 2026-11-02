"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    num_blocks = p // m  # 9 // 3 = 3
    rank_per_block = n * m * m  # 2*3*3 = 6
    total_rank = num_blocks * rank_per_block  # 3*6 = 18

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    idx = 0
    for block in range(num_blocks):
        base_col = block * m
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    U[i * m + j][idx] = 1
                    V[j * p + (base_col + k)][idx] = 1
                    W[(base_col + k) * n + i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
