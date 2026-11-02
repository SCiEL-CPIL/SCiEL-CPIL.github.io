"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12

    # Schoolbook per 2-column block: 6 blocks * 4 = rank 24
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24

    column = 0
    for block in range(6):
        k_offset = block * 2
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    U[i * m + j][column] = 1
                    V[j * p + (k_offset + k)][column] = 1
                    W[(k_offset + k) * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
