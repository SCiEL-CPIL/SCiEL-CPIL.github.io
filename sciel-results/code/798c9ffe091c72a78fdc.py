"""Matrix multiplication 4,4,9 rank-minimal decomposition.

This program constructs a rank-114 exact tensor decomposition for multiplying
a 4x4 matrix by a 4x9 matrix.

The 4x9 matrix B is split into B1 (4x4), B2 (4x4), and B3 (4x1).
The multiplication A*B is decomposed into:
1. A*B1: Uses a rank-49 bilinear algorithm for 4x4 * 4x4.
2. A*B2: Uses a rank-49 bilinear algorithm for 4x4 * 4x4.
3. A*B3: Uses the rank-16 schoolbook algorithm for 4x4 * 4x1.

The rank-49 algorithm for 4x4 * 4x4 is constructed by applying Strassen's
7-multiplication algorithm to the 2x2 block structure of the 4x4 matrices,
where each block multiplication is itself resolved using Strassen's algorithm
for 2x2 * 2x2. This yields 7 * 7 = 49 scalar multiplications.
"""

import time

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # n=4, m=4, p=9

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def zero_vec(size):
        return [0] * size

    # Helper to add a 4x4 * 4x4 rank-49 decomposition
    def add_4x4_decomp(A_offset, B_offset, C_offset):
        """
        Adds 49 terms for 4x4 * 4x4 multiplication.
        A_offset: (row, col) start index for A in global A (usually (0,0))
        B_offset: (row, col) start index for B in global B (e.g., (0,0) or (0,4))
        C_offset: (row, col) start index for C in global C (e.g., (0,0) or (0,4))
        """
        
        # 2x2 Strassen terms: (u_dict, v_dict, w_dict)
        # Keys are (row, col) in 2x2 local block
        s2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        # 4x4 Block Strassen terms: (A_blocks, B_blocks, C_blocks)
        # Each block list contains (br, bc, coeff)
        s4x4_blocks = [
            (
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)]
            ),
            (
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,0,1), (1,1,-1)]
            ),
            (
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(0,1,1), (1,1,1)]
            ),
            (
                [(1,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (1,0,1)]
            ),
            (
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,-1), (0,1,1)]
            ),
            (
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)]
            ),
            (
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)]
            ),
        ]

        for a_blocks, b_blocks, c_blocks in s4x4_blocks:
            for u_inner, v_inner, w_inner in s2x2:
                u_vec = zero_vec(n * m)
                v_vec = zero_vec(m * p)
                w_vec = zero_vec(p * n)

                for (li, lj), coeff_u in u_inner.items():
                    for (br, bc, coeff_block) in a_blocks:
                        gi = A_offset[0] + br * 2 + li
                        gj = A_offset[1] + bc * 2 + lj
                        idx = gi * m + gj
                        u_vec[idx] += coeff_u * coeff_block

                for (lj, lk), coeff_v in v_inner.items():
                    for (br, bc, coeff_block) in b_blocks:
                        gj = B_offset[0] + br * 2 + lj
                        gk = B_offset[1] + bc * 2 + lk
                        idx = gj * p + gk
                        v_vec[idx] += coeff_v * coeff_block

                for (li, lk), coeff_w in w_inner.items():
                    for (br, bc, coeff_block) in c_blocks:
                        gi = C_offset[0] + br * 2 + li
                        gk = C_offset[1] + bc * 2 + lk
                        idx = gk * n + gi
                        w_vec[idx] += coeff_w * coeff_block

                add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B1 (B1 is cols 0-3 of B)
    add_4x4_decomp((0, 0), (0, 0), (0, 0))

    # Part 2: A * B2 (B2 is cols 4-7 of B)
    add_4x4_decomp((0, 0), (0, 4), (0, 4))

    # Part 3: A * B3 (B3 is col 8 of B)
    for i in range(4):
        for j in range(4):
            u_vec = zero_vec(n * m)
            v_vec = zero_vec(m * p)
            w_vec = zero_vec(p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]

    return {"U": U, "V": V, "W": W}
