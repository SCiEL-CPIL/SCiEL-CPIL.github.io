"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Helper to set Strassen block for column pair (2k, 2k+1)
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        U[0][r1] = 1
        U[3][r1] = 1
        V[k0][r1] = 1
        V[p + k1][r1] = 1

        U[2][r2] = 1
        U[3][r2] = 1
        V[k0][r2] = 1

        U[0][r3] = 1
        V[k1][r3] = 1
        V[p + k1][r3] = -1

        U[3][r4] = 1
        V[p + k0][r4] = 1
        V[k0][r4] = -1

        U[0][r5] = 1
        U[1][r5] = 1
        V[p + k1][r5] = 1

        U[2][r6] = 1
        U[0][r6] = -1
        V[k0][r6] = 1
        V[k1][r6] = 1

        U[1][r7] = 1
        U[3][r7] = -1
        V[p + k0][r7] = 1
        V[p + k1][r7] = 1

        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1

        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1

        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1

        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1

    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    U[0][r1] = 1
    V[14][r1] = 1
    W[28][r1] = 1

    U[1][r2] = 1
    V[p + 14][r2] = 1
    W[28][r2] = 1

    U[2][r3] = 1
    V[14][r3] = 1
    W[29][r3] = 1

    U[3][r4] = 1
    V[p + 14][r4] = 1
    W[29][r4] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
