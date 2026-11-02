"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][column] = 1
            for k in range(p):
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
            column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
