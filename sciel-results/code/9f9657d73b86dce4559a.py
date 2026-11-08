"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Direct construction: split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
Use Strassen for 4x4*4x4 (rank 49 each) and schoolbook for 4x4*4x1 (rank 16).
Total rank: 49 + 49 + 16 = 114.
"""

import time


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.

    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 115  # Must finish within 120s, keep margin

    n, m, p = 4, 4, 9

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen terms for 2x2*2x2:
    # Each term: (A_coeffs, B_coeffs, C_contribs)
    strassen_meta_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    def add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
                                 A_side, B_side, C_side):
        for A_coeffs, B_coeffs, C_contribs in strassen_meta_terms:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36

            for (li, lj), meta_coeff in A_coeffs.items():
                for (br, bc, block_coeff) in A_side:
                    gi = A_base_i + br * 2 + li
                    gj = A_base_j + bc * 2 + lj
                    u_vec[gi * m + gj] += meta_coeff * block_coeff

            for (lj, lk), meta_coeff in B_coeffs.items():
                for (br, bc, block_coeff) in B_side:
                    gj = B_base_j + br * 2 + lj
                    gk = B_base_k + bc * 2 + lk
                    v_vec[gj * p + gk] += meta_coeff * block_coeff

            for (li, lk), meta_coeff in C_contribs.items():
                for (br, bc), block_coeff in C_side.items():
                    gi = C_base_i + br * 2 + li
                    gk = C_base_k + bc * 2 + lk
                    w_vec[gk * n + gi] += meta_coeff * block_coeff

            add_term(u_vec, v_vec, w_vec)

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(0, 0, 1), (1, 1, 1)],
            B_side=[(0, 0, 1), (1, 1, 1)],
            C_side={(0, 0): 1, (1, 1): 1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(1, 0, 1), (1, 1, 1)],
            B_side=[(0, 0, 1)],
            C_side={(1, 0): 1, (1, 1): -1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(0, 0, 1)],
            B_side=[(0, 1, 1), (1, 1, -1)],
            C_side={(0, 1): 1, (1, 1): 1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(1, 1, 1)],
            B_side=[(1, 0, 1), (0, 0, -1)],
            C_side={(0, 0): 1, (1, 0): 1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(0, 0, 1), (0, 1, 1)],
            B_side=[(1, 1, 1)],
            C_side={(0, 0): -1, (0, 1): 1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(1, 0, 1), (0, 0, -1)],
            B_side=[(0, 0, 1), (0, 1, 1)],
            C_side={(1, 1): 1}
        )
        add_block_multiplication(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k,
            A_side=[(0, 1, 1), (1, 1, -1)],
            B_side=[(1, 0, 1), (1, 1, 1)],
            C_side={(0, 0): 1}
        )

    if time.time() > deadline:
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u_vec = [0] * 16
                    v_vec = [0] * 36
                    w_vec = [0] * 36
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    add_term(u_vec, v_vec, w_vec)
    else:
        strassen_4x4(0, 0, 0, 0, 0, 0)

        if time.time() > deadline:
            pass

        strassen_4x4(0, 0, 0, 4, 0, 4)

        if time.time() > deadline:
            pass

        for i in range(4):
            for j in range(4):
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]

    return {"U": U, "V": V, "W": W}
