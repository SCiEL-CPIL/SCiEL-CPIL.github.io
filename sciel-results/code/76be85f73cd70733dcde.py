import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 7 * (p // 2)
    if rank < 35:
        rank = 35

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 7 products:
    # m1 = a00 * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = (a00 - a10) * (b10 + b11)
    # m4 = a11 * (b01 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a01 - a11) * (b10 + b11)
    # m7 = (a10 + a11) * (b00 - b10)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    strassen_terms = [
        # (A_row0, A_row1, B_col0, B_col1, C_col0_row0, C_col0_row1, C_col1_row0, C_col1_row1)
        # m1
        ([1, 0], [0, 1], [1, 0], [1, 0], 1, 0, 0, 1),
        # m2
        ([0, 1], [1, 1], [1, 0], [0, 0], 0, 0, 1, 0),
        # m3
        ([1, 0], [-1, 1], [0, 1], [1, 1], 0, 0, 0, 1),
        # m4
        ([0, 0], [0, 1], [1, 0], [-1, 0], 1, 0, 1, 0),
        # m5
        ([1, 1], [0, 0], [0, 0], [1, 0], -1, 0, 0, 1),
        # m6
        ([0, 1], [-1, 0], [1, 1], [0, 1], 0, 0, 0, 1),
        # m7
        ([0, 1], [1, 1], [1, 0], [0, -1], 0, 0, 0, 1),
    ]

    col = 0
    for k in range(p // 2):
        for a_r0, a_r1, b_c0, b_c1, w_c0r0, w_c0r1, w_c1r0, w_c1r1 in strassen_terms:
            # U: A is 2x2, indices [i*m+j] = [i*2+j]
            # Row 0 of A: indices 0, 1
            U[0][col] = a_r0[0]
            U[1][col] = a_r0[1]
            # Row 1 of A: indices 2, 3
            U[2][col] = a_r1[0]
            U[3][col] = a_r1[1]

            # V: B is 2x10, for block k, columns 2k and 2k+1
            # B[0, 2k] -> V[0*10 + 2k] = V[2k]
            # B[0, 2k+1] -> V[0*10 + 2k+1] = V[2k+1]
            # B[1, 2k] -> V[1*10 + 2k] = V[10+2k]
            # B[1, 2k+1] -> V[1*10 + 2k+1] = V[10+2k+1]
            V[2 * k][col] = b_c0[0]
            V[2 * k + 1][col] = b_c0[1]
            V[10 + 2 * k][col] = b_c1[0]
            V[10 + 2 * k + 1][col] = b_c1[1]

            # W: C is 2x10, convention W[k'*n + i] where k' is col, i is row
            # C[0, 2k]   -> W[2k * 2 + 0] = W[4k]
            # C[1, 2k]   -> W[2k * 2 + 1] = W[4k+1]
            # C[0, 2k+1] -> W[(2k+1)*2 + 0] = W[4k+2]
            # C[1, 2k+1] -> W[(2k+1)*2 + 1] = W[4k+3]
            W[4 * k][col] = w_c0r0
            W[4 * k + 1][col] = w_c0r1
            W[4 * k + 2][col] = w_c1r0
            W[4 * k + 3][col] = w_c1r1

            col += 1

    # Convert to list of lists for interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
