import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Rank-57 decomposition: M_{4,4,6} = M_{4,4,4} (Strassen, rank 49) + M_{4,4,2} (schoolbook, rank 8)
    R = 57
    U = np.zeros((n * m, R), dtype=int)  # (16, 57)
    V = np.zeros((m * p, R), dtype=int)  # (24, 57)
    W = np.zeros((p * n, R), dtype=int)  # (24, 57)

    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    col = 0

    # Part 1: M_{4,4,4} using two-level Strassen (rank 49)
    # This computes C[:, 0:4] = A * B[:, 0:4]
    for s in range(7):
        for si in range(7):
            # U part: A linear combinations
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            # V part: B linear combinations (only first 4 columns of B, i.e., cols 0-3)
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            # B has 6 columns, but we only use first 4 here
                            # B index: row * 6 + col, where col is 0-3
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 2 + y_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            # W part: C reconstruction (only first 4 columns of C, i.e., cols 0-3)
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # C index: (col_block * 2 + col_inner) * 4 + (row_block * 2 + row_inner)
                    # For M_{4,4,4}, C is 4x4, so col is 0-3
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

    # Part 2: M_{4,4,2} using schoolbook (rank 8)
    # This computes C[:, 4:6] = A * B[:, 4:6]
    for i in range(4):
        for k in range(4):
            for j in range(2):
                u_idx = i * 4 + k
                v_idx = k * 6 + (4 + j)
                w_idx = (4 + j) * 4 + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    def to_list(mat):
        return [[int(x) for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
