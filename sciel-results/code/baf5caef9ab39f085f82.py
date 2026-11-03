"""Matrix multiplication 4x4 * 4x6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]  # 16 x 96
    V = [[0] * rank for _ in range(m * p)]  # 24 x 96
    W = [[0] * rank for _ in range(p * n)]  # 24 x 96

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
