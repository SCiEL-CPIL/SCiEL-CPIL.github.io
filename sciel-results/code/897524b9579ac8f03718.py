import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28

    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 terms: (U_dict, V_dict, W_dict)
    # Keys are local (row, col) pairs in 2x2 blocks
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    # 4 blocks: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2),  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]

    for bi, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(blocks):
        for ti, (u_d, v_d, w_d) in enumerate(strassen):
            k = bi * 7 + ti  # column index in rank-28 matrices

            for (lr, lc), val in u_d.items():
                U[a_r + lr, a_c + lc, k] = val
            for (lr, lc), val in v_d.items():
                V[b_r + lr, b_c + lc, k] = val
            for (lr, lc), val in w_d.items():
                W[c_c + lc, c_r + lr, k] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
