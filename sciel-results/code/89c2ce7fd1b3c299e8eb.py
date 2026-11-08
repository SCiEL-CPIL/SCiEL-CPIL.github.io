import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49

    # Inner Strassen (2x2) terms: (X_coeffs, Y_coeffs)
    # Positions: 0=x00, 1=x01, 2=x10, 3=x11
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    # Inner Z contributions: (pos_in_2x2_result, coeff)
    inner_Z_contrib = [
        [(0, 1), (3, 1)],
        [(2, 1)],
        [(1, 1)],
        [(2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    # Outer Strassen (4x4 as 2x2 of 2x2 blocks)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    outer_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    # Outer C contributions: (c_block_idx, coeff)
    # C block indices: 0=C00, 1=C01, 2=C10, 3=C11
    outer_C_contrib = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)

    col = 0
    for r in range(7):
        A_outer = outer_terms[r][0]
        B_outer = outer_terms[r][1]
        C_outer = outer_C_contrib[r]

        for s in range(7):
            X_inner = inner_terms[s][0]
            Y_inner = inner_terms[s][1]
            Z_inner = inner_Z_contrib[s]

            for (bi, ca) in A_outer:
                for (pi, cx) in X_inner:
                    row4 = 2 * (bi // 2) + pi // 2
                    col4 = 2 * (bi % 2) + pi % 2
                    U[row4 * 4 + col4, col] += ca * cx

            for (bi, cb) in B_outer:
                for (pi, cy) in Y_inner:
                    row4 = 2 * (bi // 2) + pi // 2
                    col4 = 2 * (bi % 2) + pi % 2
                    V[row4 * 4 + col4, col] += cb * cy

            for (ci, cc) in C_outer:
                for (zi, cz) in Z_inner:
                    i4 = 2 * (ci // 2) + zi // 2
                    k4 = 2 * (ci % 2) + zi % 2
                    W[k4 * 4 + i4, col] += cc * cz

            col += 1

    # Verify
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for c in range(rank):
                    val += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                max_err = max(max_err, abs(val - 1.0))

    if max_err > 1e-10:
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j, col] = 1.0
                    V[j * 4 + k, col] = 1.0
                    W[k * 4 + i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
