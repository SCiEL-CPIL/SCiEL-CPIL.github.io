import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Constructs the best known rank decomposition for 2x2 by 2x10 matrix multiplication.
    Base: rank-35 via 5 independent Strassen 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def build_strassen_35():
        rank = 35
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        strassen_A = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]

        strassen_B = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]

        w_contrib = [[0]*4 for _ in range(7)]
        w_contrib[0][0] = 1; w_contrib[0][3] = 1
        w_contrib[1][2] = 1; w_contrib[1][3] = -1
        w_contrib[2][1] = 1; w_contrib[2][3] = 1
        w_contrib[3][0] = 1; w_contrib[3][2] = 1
        w_contrib[4][0] = -1; w_contrib[4][1] = 1
        w_contrib[5][3] = 1
        w_contrib[6][0] = 1

        for b in range(5):
            for r in range(7):
                col = b * 7 + r
                U[0][col] = strassen_A[r][0]
                U[1][col] = strassen_A[r][1]
                U[2][col] = strassen_A[r][2]
                U[3][col] = strassen_A[r][3]

                k0 = 2 * b
                k1 = 2 * b + 1
                V[0 * p + k0][col] = strassen_B[r][0]
                V[0 * p + k1][col] = strassen_B[r][1]
                V[1 * p + k0][col] = strassen_B[r][2]
                V[1 * p + k1][col] = strassen_B[r][3]

                W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
                W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
                W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
                W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]

        return U, V, W, rank

    U, V, W, rank = build_strassen_35()

    return {"U": U, "V": V, "W": W}
