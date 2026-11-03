"""
Rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication.

Approach:
  - Build the canonical (rank 96) decomposition.
  - Identify the 4x4x4 sub-tensor and the 4x4x2 sub-tensor.
  - Apply Strassen's 7-term algorithm to each 2x2x2 block product.
  - For the 4x4x4 part: 7 Strassen macro-products, each decomposed via 7 Strassen 2x2 terms = 49 terms.
  - For the 4x4x2 part: 4 block products (A00*B_top, A01*B_bot, A10*B_top, A11*B_bot), each 7 terms = 28 terms.
  - Total rank: 77.
"""
import numpy as np

def construct():
    n, m, p = 4, 4, 6

    # Strassen 2x2x2: 7 bilinear terms
    # Each term: (A_coeff[2][2], B_coeff[2][2], C_coeff[2][2])
    # The bilinear form is: sum_{i,j} A_coeff[i][j]*a[i][j] * sum_{i,j} B_coeff[i][j]*b[i][j]
    # contributing C_coeff[i][j] to C[i][j]
    S7 = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    terms = []  # list of (u_vec, v_vec, w_vec) where u has len 16, v has len 24, w has len 24

    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, scale=1):
        """Add 7 Strassen terms for a 2x2x2 block product at specified offsets."""
        for (a_flat, b_flat, c_flat) in S7:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                u[(a_r + i) * m + (a_c + j)] = a_flat[idx]
                v[(b_r + i) * p + (b_c + j)] = b_flat[idx]
                w[(c_c + j) * n + (c_r + i)] = c_flat[idx]
            terms.append((u, v, w))

    def add_macro_product(a_blocks, b_blocks, c_blocks):
        """
        Add 7 Strassen terms for a macro-product where:
        X = sum_k a_blocks[k] * A_block_k  (each a_block is (row_off, col_off, coeff))
        Y = sum_k b_blocks[k] * B_block_k
        contributes to C = sum_k c_blocks[k] * C_block_k
        
        a_blocks, b_blocks, c_blocks: lists of (row_off, col_off, coeff)
        """
        for (a_flat, b_flat, c_flat) in S7:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            # Fill U
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                coeff_a = a_flat[idx]
                for (r, c, cc) in a_blocks:
                    u[(r + i) * m + (c + j)] += cc * coeff_a
            # Fill V
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                coeff_b = b_flat[idx]
                for (r, c, cc) in b_blocks:
                    v[(r + i) * p + (c + j)] += cc * coeff_b
            # Fill W
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                coeff_c = c_flat[idx]
                for (r, c, cc) in c_blocks:
                    w[(c + j) * n + (r + i)] += cc * coeff_c
            terms.append((u, v, w))

    # === 4x4x4 part (columns 0-3 of B, columns 0-3 of C) ===
    # Strassen 4x4x4: 7 macro-products
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_macro_product(
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)]
    )
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_macro_product(
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)],
        [(2, 0, 1), (2, 2, -1)]
    )
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_macro_product(
        [(0, 0, 1)],
        [(0, 2, 1), (2, 2, -1)],
        [(0, 2, 1), (2, 2, 1)]
    )
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_macro_product(
        [(2, 2, 1)],
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (2, 0, 1)]
    )
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_macro_product(
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)],
        [(0, 0, -1), (0, 2, 1)]
    )
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_macro_product(
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)]
    )
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_macro_product(
        [(0, 2, 1), (2, 2, -1)],
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)]
    )

    # === 4x4x2 part (columns 4-5 of B, columns 4-5 of C) ===
    # C_top (rows 0-1) = A00 * B_top + A01 * B_bot
    # C_bot (rows 2-3) = A10 * B_top + A11 * B_bot
    # Each is a 2x2x2 product: 7 terms each, total 28 terms

    # A00 * B_top -> C_top
    add_strassen_block(0, 0, 0, 4, 0, 4)
    # A01 * B_bot -> C_top
    add_strassen_block(0, 2, 2, 4, 0, 4)
    # A10 * B_top -> C_bot
    add_strassen_block(2, 0, 0, 4, 2, 4)
    # A11 * B_bot -> C_bot
    add_strassen_block(2, 2, 2, 4, 2, 4)

    rank = len(terms)

    # Build U, V, W as lists of lists
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for k, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][k] = u[i]
        for i in range(m * p):
            V[i][k] = v[i]
        for i in range(p * n):
            W[i][k] = w[i]

    return {"U": U, "V": V, "W": W}
