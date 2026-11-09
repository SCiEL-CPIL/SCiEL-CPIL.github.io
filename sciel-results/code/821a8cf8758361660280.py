"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 6 x 27
    V = [[0] * rank for _ in range(m * p)]  # 27 x 27
    W = [[0] * rank for _ in range(p * n)]  # 18 x 27

    for i in range(n):  # 0, 1
        for j in range(m):  # 0, 1, 2
            for k in range(p):  # 0..8
                c = i * 9 + j * 3 + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
