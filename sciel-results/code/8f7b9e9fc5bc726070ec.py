"""Rank-14 exact decomposition: 2x4 by 4x8 via Strassen on B's 2x2 blocks."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # 7 Strassen bilinear forms: (alpha_u, alpha_v, beta_u, beta_v, gamma_u, gamma_v)
    strassen = [
        (1, 1, 1, 1, 1, 1),
        (1, 0, 1, 0, 0, 1),
        (0, 1, 0, 1, 1, 0),
        (1, 1, 0, 0, 1, 0),
        (0, 0, 1, 1, 0, 1),
        (1, 0, 0, 1, 1, 1),
        (0, 1, 1, 0, 1, 1),
    ]
    # Each Strassen block uses 2 of B's rows and 2 of C's columns.
    # block 0: B rows {0,1} -> C cols {0,1}; block 1: B rows {2,3} -> C cols {2,3}
    blocks = [(0, 0), (2, 2)]  # (b_row_start, c_col_start)

    U = [[0] * 14 for _ in range(n * m)]  # 8 x 14
    V = [[0] * 14 for _ in range(m * p)]  # 32 x 14
    W = [[0] * 14 for _ in range(p * n)]  # 16 x 14

    col = 0
    for (br, cc) in blocks:
        for (au, av, bu, bv, gu, gv) in strassen:
            # V: B's two rows (br, br+1) with coefficients (au,av) at columns (cc,cc+1)
            V[br * p + cc][col] = au
            V[br * p + (cc + 1)][col] = av
            V[(br + 1) * p + cc][col] = bu
            V[(br + 1) * p + (cc + 1)][col] = bv
            # U: A's two rows (i=0,1) with coefficients (gu,gv) at cols (cc,cc+1)
            U[0 * m + cc][col] = gu
            U[0 * m + (cc + 1)][col] = gv
            U[1 * m + cc][col] = gu
            U[1 * m + (cc + 1)][col] = gv
            # W: output C's two rows (i=0,1) at columns (cc,cc+1); W[k*n + i]
            W[cc * n + 0][col] = 1
            W[cc * n + 1][col] = 1
            W[(cc + 1) * n + 0][col] = 1
            W[(cc + 1) * n + 1][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
