import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for b in range(5):
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 11 + 2 * b
        v11 = 12 + 2 * b
        w00 = 4 * b
        w01 = 4 * b + 2
        w10 = 4 * b + 1
        w11 = 4 * b + 3

        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1
        r += 1

        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] -= 1
        r += 1

        U[0][r] += 1
        V[v01][r] += 1; V[v11][r] -= 1
        W[w01][r] += 1; W[w11][r] += 1
        r += 1

        U[3][r] += 1
        V[v10][r] += 1; V[v00][r] -= 1
        W[w00][r] += 1; W[w10][r] += 1
        r += 1

        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1; W[w01][r] += 1
        r += 1

        U[2][r] += 1; U[0][r] -= 1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1
        r += 1

        U[1][r] += 1; U[3][r] -= 1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1
        r += 1

    U[0][r] += 1; V[10][r] += 1; W[20][r] += 1; r += 1
    U[1][r] += 1; V[21][r] += 1; W[20][r] += 1; r += 1
    U[2][r] += 1; V[10][r] += 1; W[21][r] += 1; r += 1
    U[3][r] += 1; V[21][r] += 1; W[21][r] += 1; r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
