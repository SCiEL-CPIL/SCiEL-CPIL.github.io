"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for l in range(5):
        k1 = 2 * l
        k2 = 2 * l + 1
        base = 7 * l

        U[0][base] = 1
        U[3][base] = 1
        V[k1][base] = 1
        V[10 + k2][base] = 1
        W[k1][base] = 1
        W[k2 + 2][base] = 1

        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k1][base + 1] = 1
        W[k1 + 2][base + 1] = 1
        W[k2 + 2][base + 1] = -1

        U[0][base + 2] = 1
        V[k2][base + 2] = 1
        V[10 + k2][base + 2] = -1
        W[k2][base + 2] = 1
        W[k2 + 2][base + 2] = 1

        U[3][base + 3] = 1
        V[10 + k1][base + 3] = 1
        V[k1][base + 3] = -1
        W[k1][base + 3] = 1
        W[k1 + 2][base + 3] = 1

        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[10 + k2][base + 4] = 1
        W[k1][base + 4] = -1
        W[k2][base + 4] = 1

        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k1][base + 5] = 1
        V[k2][base + 5] = 1
        W[k2 + 2][base + 5] = 1

        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[10 + k1][base + 6] = 1
        V[10 + k2][base + 6] = 1
        W[k1][base + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
