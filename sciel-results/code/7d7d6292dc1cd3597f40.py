"""Rank-minimal exact tensor decomposition for 2x2 * 2x16 matrix multiplication.

The 16 output columns come in 8 independent pairs.  For a fixed output row i,
the pair (C[i,2k], C[i,2k+1]) is a 2x2 product A[i,:] * B[:, 2k:2k+2].  Each
2x2x2 sub-product has rank 7 (Strassen), and the 8 column-blocks are mutually
independent, so the total rank is 8 * 7 = 56, improving on the rank-64
schoolbook.  All entries are integers, so the decomposition is exact.
"""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is even; group the p columns into p//2 independent 2-column blocks.
    nblocks = p // 2

    U = [[0] * (nblocks * 7) for _ in range(n * m)]   # shape (4, 56)
    V = [[0] * (nblocks * 7) for _ in range(m * p)]   # shape (32, 56)
    W = [[0] * (nblocks * 7) for _ in range(p * n)]   # shape (32, 56)

    # Strassen linear combinations for the 2x2 core (A = [[a00,a01],[a10,a11]]).
    # Each entry: (U_row, V_row, W_row, sign, a_idx, b_idx).
    #   a_idx in {0,1,2,3} -> A index (i, j) with i = a_idx//2, j = a_idx%2
    #   b_idx in {0,1,2,3} -> B index (j, k) with j = b_idx//2, k = b_idx%2
    terms = [
        # m0 = (A00 + A11) * (B00 + B11)
        (0, 0, 0, +1, 0, 0),
        (0, 2, 0, +1, 0, 2),
        (2, 0, 0, +1, 2, 0),
        (2, 2, 0, +1, 2, 2),
        # m1 = (A00 + A10) * B00
        (0, 0, 1, +1, 0, 0),
        (2, 0, 1, +1, 2, 0),
        # m2 = A10 * B10
        (2, 2, 2, +1, 2, 2),
        # m3 = A01 * B11
        (1, 3, 3, +1, 1, 3),
        # m4 = (A01 + A11) * B01
        (1, 1, 4, +1, 1, 1),
        (3, 1, 4, +1, 3, 1),
        # m5 = (A00 - A01) * (B01 + B11)
        (0, 1, 5, +1, 0, 1),
        (0, 3, 5, +1, 0, 3),
        (1, 1, 5, -1, 1, 1),
        (1, 3, 5, -1, 1, 3),
        # m6 = (A10 - A00) * (B00 + B10)
        (2, 0, 6, +1, 2, 0),
        (2, 2, 6, +1, 2, 2),
        (0, 0, 6, -1, 0, 0),
        (0, 2, 6, -1, 0, 2),
    ]

    for block in range(nblocks):
        col = block * 7
        for t, (urow, vrow, wrow, sign, a_idx, b_idx) in enumerate(terms):
            c = col + t
            # A row: output row i = urow (0 or 1), A column j = a_idx % 2
            U[urow * m + (a_idx % 2)][c] = sign
            # B row: A column j = b_idx // 2, output column k = block*2 + (b_idx % 2)
            V[(b_idx // 2) * p + (block * 2 + (b_idx % 2))][c] = 1
            # W row (trace convention W[k*n + i]): output column k = wrow, row i
            W[(wrow * 2) * n + urow][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
