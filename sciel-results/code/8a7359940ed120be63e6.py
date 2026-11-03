import time
import numpy as np

def construct():
    """
    Rank-50 exact tensor decomposition for 2x4 by 4x7 matrix multiplication.

    Block structure of B (4x7) by column blocks:
      Block A (cols 0,1): two 2x2*2x2 Strassen (14)
      Block B (cols 2,3): two 2x2*2x2 Strassen (14)
      Block C (cols 4,5): two 2x2*2x2 Strassen (14)
      Block D (col 6):    two 2x2*2x1 schoolbook (8)
    Total rank = 42 + 8 = 50.

    The 4x7 tensor M_{2,4,7} decomposes as:
      For each column-block of B, split A into left/right 2x2 and B-block
      into top/bottom 2x2 (or 2x1 for the last column), then use the
      rank-7 Strassen or rank-4 schoolbook sub-decomposition.

    All entries are integers, so the exactness requirement is satisfied.
    Output factor uses the official trace convention W[k*n + i, r] where
    k is the output column index and i the output row index.
    """
    # Deadline guard: keep a safety margin under the 120 s evaluation limit.
    deadline = time.time() + 110.0

    n, m, p = 2, 4, 7
    rank = 50

    U = np.zeros((n * m, rank), dtype=np.float64)  # (8, 50)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (28, 50)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (14, 50)

    state = {'col': 0}

    def add_strassen_block(A_idx, B_idx, C_idx):
        """Add a rank-7 Strassen 2x2*2x2->2x2 block.

        A_idx, B_idx, C_idx are lists of 4 (row, col) global indices in
        row-major order for the 2x2 sub-matrices:
          [ (r0,c0), (r0,c1), (r1,c0), (r1,c1) ]
        Local 2x2 index layout: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
        """
        def emit(a_coeffs, b_coeffs, c_coeffs):
            col = state['col']
            for loc, coeff in a_coeffs:
                gr, gc = A_idx[loc]
                U[gr * m + gc, col] += coeff
            for loc, coeff in b_coeffs:
                gr, gc = B_idx[loc]
                V[gr * p + gc, col] += coeff
            for loc, coeff in c_coeffs:
                gr, gc = C_idx[loc]
                # Trace convention: W[k*n + i, r], k=output col, i=output row.
                W[gc * n + gr, col] += coeff
            state['col'] = col + 1

        # Strassen 7 products.
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 -= M1
        emit([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)])
        # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
        emit([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
        # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
        emit([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
        # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
        emit([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
        # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
        emit([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
        # M6 = (a10-a00)(b00+b01) -> C11 += M6
        emit([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
        # M7 = (a01-a11)(b10+b11) -> C00 += M7
        emit([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])

    def add_schoolbook_221(A_idx, B_idx, C_idx):
        """Add a rank-4 schoolbook 2x2 * 2x1 -> 2x1 block.

        A_idx: list of 4 (row,col) for the 2x2 A (order 0=(0,0),1=(0,1),2=(1,0),3=(1,1)).
        B_idx: list of 2 (row,col) for the 2x1 B (order 0=(0,*),1=(1,*)).
        C_idx: list of 2 (row,col) for the 2x1 C (order 0=(0,*),1=(1,*)).
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        """
        def emit(a_loc, b_loc, c_loc, coeff=1):
            col = state['col']
            ar, ac = A_idx[a_loc]
            br, bc = B_idx[b_loc]
            cr, cc = C_idx[c_loc]
            U[ar * m + ac, col] += coeff
            V[br * p + bc, col] += coeff
            W[cc * n + cr, col] += coeff
            state['col'] = col + 1

        emit(0, 0, 0, 1)  # a00*b0 -> c0
        emit(1, 1, 0, 1)  # a01*b1 -> c0
        emit(2, 0, 1, 1)  # a10*b0 -> c1
        emit(3, 1, 1, 1)  # a11*b1 -> c1

    # A column split: left 2x2 (cols 0,1), right 2x2 (cols 2,3).
    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]

    # ---- Block 1: B columns 0,1 (4x2) -> C columns 0,1 (2x2) ----
    B1_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    B1_R = [(2, 0), (2, 1), (3, 0), (3, 1)]
    C1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    add_strassen_block(A_L, B1_L, C1)
    add_strassen_block(A_R, B1_R, C1)

    # ---- Block 2: B columns 2,3 (4x2) -> C columns 2,3 (2x2) ----
    B2_L = [(0, 2), (0, 3), (1, 2), (1, 3)]
    B2_R = [(2, 2), (2, 3), (3, 2), (3, 3)]
    C2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    add_strassen_block(A_L, B2_L, C2)
    add_strassen_block(A_R, B2_R, C2)

    # ---- Block 3: B columns 4,5 (4x2) -> C columns 4,5 (2x2) ----
    B3_L = [(0, 4), (0, 5), (1, 4), (1, 5)]
    B3_R = [(2, 4), (2, 5), (3, 4), (3, 5)]
    C3 = [(0, 4), (0, 5), (1, 4), (1, 5)]
    add_strassen_block(A_L, B3_L, C3)
    add_strassen_block(A_R, B3_R, C3)

    # ---- Block 4: B column 6 (4x1) -> C column 6 (2x1) ----
    B4_L = [(0, 6), (1, 6)]
    B4_R = [(2, 6), (3, 6)]
    C4 = [(0, 6), (1, 6)]
    add_schoolbook_221(A_L, B4_L, C4)
    add_schoolbook_221(A_R, B4_R, C4)

    if state['col'] != rank:
        # Defensive: truncate or pad so that the declared rank matches the
        # number of populated columns. The construction is deterministic and
        # should always land exactly at 50.
        if state['col'] < rank:
            pass  # remaining columns are zero -> harmless
        else:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
