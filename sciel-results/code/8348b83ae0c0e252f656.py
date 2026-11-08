"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]

STRASSEN_2X2 = [
    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
]

BLOCK_STRASSEN = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
    ([(1,0,1),(1,1,1)], [(0,0,1)],       {(1,0):1, (1,1):-1}),
    ([(0,0,1)],       [(0,1,1),(1,1,-1)],{(0,1):1, (1,1):1}),
    ([(1,1,1)],       [(1,0,1),(0,0,-1)],{(0,0):1, (1,0):1}),
    ([(0,0,1),(0,1,1)],[(1,1,1)],        {(0,0):-1,(0,1):1}),
    ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)],{(1,1):1}),
    ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)],{(0,0):1}),
]


def construct():
    n, m, p = DIMENSIONS
    num_A = n * m
    num_B = m * p
    num_C = p * n

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        for A_side, B_side, C_side in BLOCK_STRASSEN:
            for A_meta, B_meta, C_meta in STRASSEN_2X2:
                u_vec = [0] * num_A
                v_vec = [0] * num_B
                w_vec = [0] * num_C

                for (li, lj), mc in A_meta.items():
                    for (br, bc, bc_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += mc * bc_coeff

                for (lj, lk), mc in B_meta.items():
                    for (br, bc, bc_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += mc * bc_coeff

                for (li, lk), mc in C_meta.items():
                    for (br, bc), bc_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += mc * bc_coeff

                add_term(u_vec, v_vec, w_vec)

    strassen_4x4(0, 0, 0, 0, 0, 0)
    strassen_4x4(0, 0, 0, 4, 0, 4)

    for i in range(4):
        for j in range(4):
            u_vec = [0] * num_A
            v_vec = [0] * num_B
            w_vec = [0] * num_C
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(num_A)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(num_B)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(num_C)]

    return {"U": U, "V": V, "W": W}
