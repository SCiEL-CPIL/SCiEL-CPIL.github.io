import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39: 4 Strassen 2x2 blocks (cols 0-7) + 2x3 schoolbook+strassen (cols 8-10)
    # Block 5 (cols 8,9): Strassen rank 7
    # Col 10: schoolbook rank 4
    # Total: 4*7 + 7 + 4 = 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def strassen_block(b, r0):
        # Strassen for A(2x2) * B[:, 2b:2b+2] -> C[:, 2b:2b+2]
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 11 + 2 * b
        v11 = 12 + 2 * b
        w00 = 4 * b
        w01 = 4 * b + 2
        w10 = 4 * b + 1
        w11 = 4 * b + 3

        r = r0
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
        return r

    r = 0
    for b in range(5):
        r = strassen_block(b, r)

    # Col 10 schoolbook: 4 terms
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    U[0][r] = 1; V[10][r] = 1; W[20][r] = 1
    r += 1
    U[1][r] = 1; V[21][r] = 1; W[20][r] = 1
    r += 1
    U[2][r] = 1; V[10][r] = 1; W[21][r] = 1
    r += 1
    U[3][r] = 1; V[21][r] = 1; W[21][r] = 1
    r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
