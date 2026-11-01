"""Program for 2x3x7 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # We use a hybrid decomposition:
    # 1. Split m=3 into m1=2, m2=1.
    # 2. A = [A1, A2], B = [B1; B2].
    # 3. C = A1*B1 + A2*B2.
    # 4. A1 is 2x2, B1 is 2x7.
    #    Split p=7 into p1=4, p2=2, p3=1.
    #    B1 = [B11, B12, B13].
    #    C1 = A1*B11 + A1*B12 + A1*B13.
    #    A1*B11 (2x2x4): Split p=4 into 2+2. Use Strassen for 2x2x2 twice.
    #    A1*B12 (2x2x2): Use Strassen.
    #    A1*B13 (2x2x1): Use Schoolbook.
    # 5. A2 is 2x1, B2 is 1x7.
    #    Use Schoolbook.

    # We will build the U, V, W matrices by appending columns for each sub-decomposition.
    U_cols = []
    V_cols = []
    W_cols = []

    def add_decomp(u_vec, v_vec, w_vec):
        """
        u_vec: list of (index, coeff) for U (size n*m)
        v_vec: list of (index, coeff) for V (size m*p)
        w_vec: list of (index, coeff) for W (size p*n)
        """
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)

        for idx, coeff in u_vec:
            U_col[idx] = coeff
        for idx, coeff in v_vec:
            V_col[idx] = coeff
        for idx, coeff in w_vec:
            W_col[idx] = coeff

        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Helper to create index for U (A matrix, n x m)
    # U index: i * m + j
    def u_idx(i, j):
        return i * m + j

    # Helper to create index for V (B matrix, m x p)
    # V index: j * p + k
    def v_idx(j, k):
        return j * p + k

    # Helper to create index for W (C matrix, n x p)
    # W index: k * n + i
    def w_idx(k, i):
        return k * n + i

    # --- Part 1: A2 * B2 (2x1 x 7) ---
    for i in range(n):
        for k in range(p):
            # Term: A[i, 2] * B[2, k]
            u_vec = [(u_idx(i, 2), 1.0)]
            v_vec = [(v_idx(2, k), 1.0)]
            w_vec = [(w_idx(k, i), 1.0)]
            add_decomp(u_vec, v_vec, w_vec)

    # --- Part 2: A1 * B1 (2x2 x 7) ---
    # A1 is A[:, 0:2]. B1 is B[0:2, :].
    # Split B1 into B11 (cols 0:4), B12 (cols 4:6), B13 (cols 6:7).

    def strassen_decomp(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        """
        Adds Strassen decomposition for A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        times B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        to C[a_row_offset:a_row_offset+2, b_col_offset:b_col_offset+2]
        """
        # M1 = (A00+A11)(B00+B11)
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)

        # M2 = (A10+A11)B00
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1), (w_idx(b_col_offset+1, a_row_offset+1), -1)]
        add_decomp(u, v, w)

        # M3 = A00(B01-B11)
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+1), 1), (v_idx(b_row_offset+1, b_col_offset+1), -1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)

        # M4 = A11(B10-B00)
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+0), -1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)

        # M5 = (A00+A01)B11
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), -1), (w_idx(b_col_offset+1, a_row_offset+0), 1)]
        add_decomp(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+0), -1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1), (u_idx(a_row_offset+1, a_col_offset+1), -1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

    def schoolbook_221(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # A is 2x2, B is 2x1.
        # C is 2x1.
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10

        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)

        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)

    # A1 is at a_row_offset=0, a_col_offset=0.
    # B1 is at b_row_offset=0.

    strassen_decomp(0, 0, 0, 0)
    strassen_decomp(0, 0, 0, 2)
    strassen_decomp(0, 0, 0, 4)
    schoolbook_221(0, 0, 0, 6)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
