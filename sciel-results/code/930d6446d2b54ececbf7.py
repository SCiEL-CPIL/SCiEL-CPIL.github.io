import numpy as np
import time

def construct():
    t0 = time.time()
    n = m = p = 6

    # Strassen 2x2 base case: rank 7
    # U_t: linear forms on A (6x6 split into 3x3 blocks of 2x2)
    # V_t: linear forms on B
    # W_t: linear forms for C reconstruction

    # Define Strassen 2x2 via index patterns (row, col, coeff)
    s2u = (
        ((0,0,1),(1,1,1)),
        ((1,0,1),(1,1,1)),
        ((0,0,1),),
        ((1,1,1),),
        ((0,0,1),(0,1,1)),
        ((1,0,1),(0,0,-1)),
        ((0,1,1),(1,1,-1)),
    )
    s2v = (
        ((0,0,1),(1,1,1)),
        ((0,0,1),),
        ((0,1,1),(1,1,-1)),
        ((1,0,1),(0,0,-1)),
        ((1,1,1),),
        ((0,0,1),(0,1,1)),
        ((1,0,1),(1,1,1)),
    )
    s2w = (
        ((0,0,1),(1,1,1)),
        ((1,0,1),(1,1,-1)),
        ((0,1,1),(1,1,1)),
        ((0,0,1),(1,0,1)),
        ((0,0,-1),(0,1,1)),
        ((1,1,1),),
        ((0,0,1),),
    )

    # Build 6x6x6 rank-189 tensor via 3x3 tiling of 2x2 Strassen
    R = 27 * 7
    U = np.zeros((36, R), dtype=np.float64)
    V = np.zeros((36, R), dtype=np.float64)
    W = np.zeros((36, R), dtype=np.float64)

    # Iterate over 3x3 grid of 2x2 blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 9 + bj * 3 + bk) * 7
                for t in range(7):
                    c = base_col + t
                    # U part: A block at (bi, bj)
                    for (dr, dc, cf) in s2u[t]:
                        U[bi * 2 + dr, bj * 2 + dc] = U[bi * 2 + dr, bj * 2 + dc] + cf * 1.0
                    # Actually U rows are indexed by (row_of_A, col_of_A) = (i*m+j)
                    # Let me fix: A is 6x6, row = bi*2+dr, col = bj*2+dc -> flat = (bi*2+dr)*6 + (bj*2+dc)
                    pass

    # Redo properly with correct flat indexing
    U = np.zeros((36, R), dtype=np.float64)
    V = np.zeros((36, R), dtype=np.float64)
    W = np.zeros((36, R), dtype=np.float64)

    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 9 + bj * 3 + bk) * 7
                for t in range(7):
                    c = base_col + t
                    # U: A[row, col] where row = bi*2+dr, col = bj*2+dc
                    for (dr, dc, cf) in s2u[t]:
                        r = (bi * 2 + dr) * 6 + (bj * 2 + dc)
                        U[r, c] += cf
                    # V: B[row, col] where row = bj*2+dr, col = bk*2+dc
                    for (dr, dc, cf) in s2v[t]:
                        r = (bj * 2 + dr) * 6 + (bk * 2 + dc)
                        V[r, c] += cf
                    # W: C[row, col] where row = bk*2+dc, col = bi*2+dr
                    # W index: [k*n + i, r] per problem statement
                    # C[row, col]: row = bk*2+dc (output row), col = bi*2+dr (output col)
                    # W flat index = (output_row) * n + output_col = (bk*2+dc)*6 + (bi*2+dr)
                    for (dr, dc, cf) in s2w[t]:
                        r = (bk * 2 + dc) * 6 + (bi * 2 + dr)
                        W[r, c] += cf

    if time.time() - t0 > 130:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
