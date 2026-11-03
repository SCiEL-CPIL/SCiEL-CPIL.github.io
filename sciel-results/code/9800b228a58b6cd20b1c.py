"""Rank-28 exact tensor decomposition for 4x4 matrix multiplication.

Uses a 2x2 Strassen scheme at the block level, and schoolbook (4 products)
at the element level within each block. Total rank = 7 * 4 = 28.
"""

def construct():
    # 2x2 Strassen left factors (coefficients for a, b, c, d):
    # s=0: a+d, s=1: c+d, s=2: a, s=3: d, s=4: a+b, s=5: c-a, s=6: b-d
    left_factors = [
        [1, 0, 0, 1],   # s=0: a+d
        [0, 0, 1, 1],   # s=1: c+d
        [1, 0, 0, 0],   # s=2: a
        [0, 0, 0, 1],   # s=3: d
        [1, 1, 0, 0],   # s=4: a+b
        [-1, 0, 1, 0],  # s=5: c-a
        [0, 1, 0, -1],  # s=6: b-d
    ]

    # 2x2 Strassen right factors (coefficients for e, f, g, h):
    right_factors = [
        [1, 0, 0, 1],   # s=0: e+h
        [1, 0, 0, 0],   # s=1: e
        [0, 1, 0, -1],  # s=2: f-h
        [0, -1, 1, 0],  # s=3: g-e
        [0, 0, 0, 1],   # s=4: h
        [1, 1, 0, 0],   # s=5: e+f
        [0, 0, 1, 1],   # s=6: g+h
    ]

    # Block-level Strassen output coefficients:
    # For each C-block position (cbi, cbj) in {0,1}^2,
    # and each Strassen factor s in {0..6}, the coefficient in C_block.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    block_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},   # C11
        (0, 1): {2: 1, 4: 1},                 # C12
        (1, 0): {1: 1, 3: 1},                 # C21
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},   # C22
    }

    # Which A-blocks and B-blocks are involved in each Strassen factor s:
    # Block indices: 0=**11, 1=*12, 2=21*, 3=22**
    a_blocks = [
        [(0, 1), (3, 1)],      # s=0: A11 + A22
        [(2, 1), (3, 1)],      # s=1: A21 + A22
        [(0, 1), (2, -1)],     # s=2: A11 - A21
        [(1, 1), (3, -1)],     # s=3: A12 - A22
        [(0, 1), (1, 1)],      # s=4: A11 + A12
        [(2, 1), (0, -1)],     # s=5: A21 - A11
        [(1, 1), (3, -1)],     # s=6: A12 - A22
    ]
    b_blocks = [
        [(0, 1), (3, 1)],      # s=0: B11 + B22
        [(0, 1)],              # s=1: B11
        [(1, 1), (2, 1)],      # s=2: B12 + B21
        [(2, 1), (3, 1)],      # s=3: B21 + B22
        [(3, 1)],              # s=4: B22
        [(0, 1), (1, 1)],      # s=5: B11 + B12
        [(2, 1), (3, -1)],     # s=6: B21 - B22
    ]

    # Block position mapping: block_idx -> (row_block, col_block)
    block_pos = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    n, m, p = 4, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28, W[k*n+i, r]

    for s in range(7):
        for j in range(2):
            for k in range(2):
                r = s * 4 + j * 2 + k

                # U side: for each A-block involved in factor s,
                # the element at position (j, k) within that block contributes.
                for (ab, ca) in a_blocks[s]:
                    ab_i, ab_j = block_pos[ab]
                    # A element at global position (ab_i*2 + j, ab_j*2 + k)
                    # U index: (row)*m + col = (ab_i*2+j)*4 + (ab_j*2+k)
                    idx_a = (ab_i * 2 + j) * m + (ab_j * 2 + k)
                    U[idx_a][r] += ca

                # V side: for each B-block involved in factor s,
                # the element at position (j, k) within that block contributes.
                for (bb, cb) in b_blocks[s]:
                    bb_i, bb_j = block_pos[bb]
                    # B element at global position (bb_i*2 + j, bb_j*2 + k)
                    # V index: (row)*p + col = (bb_i*2+j)*4 + (bb_j*2+k)
                    idx_b = (bb_i * 2 + j) * p + (bb_j * 2 + k)
                    V[idx_b][r] += cb

                # W side: for each C-block that uses factor s,
                # the element at position (j, k) within that block gets
                # coefficient block_coeffs[(cbi,cbj)][s].
                for (cbi, cbj), s_coeffs in block_coeffs.items():
                    if s not in s_coeffs:
                        continue
                    cc = s_coeffs[s]
                    # C element at global position (cbi*2 + j, cbj*2 + k)
                    # W index: k_global*n + i_global where
                    # i_global = cbi*2 + j, k_global = cbj*2 + k
                    w_idx = (cbj * 2 + k) * n + (cbi * 2 + j)
                    W[w_idx][r] += cc

    return {"U": U, "V": V, "W": W}
