"""Matrix multiplication 2,2,12 - rank-42 Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm for each 2x2x2 block (6 blocks total).
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    num_blocks = p // 2  # 6 blocks of width 2

    rank = num_blocks * 7  # 42

    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42

    # Strassen factors for 2x2x2
    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            c = base + s

            v_b00 = 0 * p + 2 * t
            v_b01 = 0 * p + 2 * t + 1
            v_b10 = 1 * p + 2 * t
            v_b11 = 1 * p + 2 * t + 1

            w_c00 = (2 * t) * n + 0
            w_c01 = (2 * t + 1) * n + 0
            w_c10 = (2 * t) * n + 1
            w_c11 = (2 * t + 1) * n + 1

            if s == 0:  # M1
                U[0][c] = 1
                U[3][c] = 1
                V[v_b00][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = 1
                W[w_c11][c] = 1

            elif s == 1:  # M2
                U[2][c] = 1
                U[3][c] = 1
                V[v_b00][c] = 1
                W[w_c10][c] = 1
                W[w_c11][c] = -1

            elif s == 2:  # M3
                U[0][c] = 1
                V[v_b01][c] = 1
                V[v_b11][c] = -1
                W[w_c01][c] = 1
                W[w_c11][c] = 1

            elif s == 3:  # M4
                U[3][c] = 1
                V[v_b10][c] = 1
                V[v_b00][c] = -1
                W[w_c00][c] = 1
                W[w_c10][c] = 1

            elif s == 4:  # M5
                U[0][c] = 1
                U[1][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = -1
                W[w_c01][c] = 1

            elif s == 5:  # M6
                U[2][c] = 1
                U[0][c] = -1
                V[v_b00][c] = 1
                V[v_b01][c] = 1
                W[w_c11][c] = 1

            elif s == 6:  # M7
                U[1][c] = 1
                U[3][c] = -1
                V[v_b10][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
