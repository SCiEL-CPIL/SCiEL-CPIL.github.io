import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1
        U[X_rows[0], col_offset + 0] += 1
        U[X_rows[3], col_offset + 0] += 1
        V[Y_rows[0], col_offset + 0] += 1
        V[Y_rows[3], col_offset + 0] += 1
        W[Z_rows[0], col_offset + 0] += 1
        W[Z_rows[3], col_offset + 0] += 1

        # M2
        U[X_rows[2], col_offset + 1] += 1
        U[X_rows[3], col_offset + 1] += 1
        V[Y_rows[0], col_offset + 1] += 1
        W[Z_rows[2], col_offset + 1] += 1
        W[Z_rows[3], col_offset + 1] -= 1

        # M3
        U[X_rows[0], col_offset + 2] += 1
        V[Y_rows[1], col_offset + 2] += 1
        V[Y_rows[3], col_offset + 2] -= 1
        W[Z_rows[1], col_offset + 2] += 1
        W[Z_rows[3], col_offset + 2] += 1

        # M4
        U[X_rows[3], col_offset + 3] += 1
        V[Y_rows[2], col_offset + 3] += 1
        V[Y_rows[0], col_offset + 3] -= 1
        W[Z_rows[0], col_offset + 3] += 1
        W[Z_rows[2], col_offset + 3] += 1

        # M5
        U[X_rows[0], col_offset + 4] += 1
        U[X_rows[1], col_offset + 4] += 1
        V[Y_rows[3], col_offset + 4] += 1
        W[Z_rows[0], col_offset + 4] -= 1
        W[Z_rows[1], col_offset + 4] += 1

        # M6
        U[X_rows[2], col_offset + 5] += 1
        U[X_rows[0], col_offset + 5] -= 1
        V[Y_rows[0], col_offset + 5] += 1
        V[Y_rows[1], col_offset + 5] += 1
        W[Z_rows[3], col_offset + 5] += 1

        # M7
        U[X_rows[1], col_offset + 6] += 1
        U[X_rows[3], col_offset + 6] -= 1
        V[Y_rows[2], col_offset + 6] += 1
        V[Y_rows[3], col_offset + 6] += 1
        W[Z_rows[0], col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
