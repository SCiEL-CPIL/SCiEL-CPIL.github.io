import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 coefficients
    # s: 0..6
    # S_X[s][pq] where pq in 0..3 (00, 01, 10, 11)
    S_X = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    S_Y = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    W_str = [
        [[1, 0], [0, 1]], # m1
        [[0, 1], [0, -1]], # m2
        [[0, 1], [0, 1]], # m3
        [[1, 0], [0, 1]], # m4
        [[-1, 0], [1, 0]], # m5
        [[0, 0], [1, 0]], # m6
        [[1, -1], [0, 0]]  # m7
    ]

    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for I in range(3):
        for J in range(3):
            for K in range(3):
                base_col = (I * 9 + J * 3 + K) * 7
                for s in range(7):
                    c = base_col + s
                    # Fill U
                    for i in range(2):
                        for j in range(2):
                            row_u = (I * 2 + i) * 6 + (J * 2 + j)
                            U[row_u][c] = S_X[s][i * 2 + j]
                            
                    # Fill V
                    for j in range(2):
                        for k in range(2):
                            row_v = (J * 2 + j) * 6 + (K * 2 + k)
                            V[row_v][c] = S_Y[s][j * 2 + k]
                            
                    # Fill W
                    for k in range(2):
                        for i in range(2):
                            row_w = (K * 2 + k) * 6 + (I * 2 + i)
                            W[row_w][c] = W_str[s][k][i]

    return {"U": U, "V": V, "W": W}
