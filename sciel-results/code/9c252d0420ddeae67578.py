"""Program for 2x3x7 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Hybrid decomposition:
    # Split m=3 into m1=2, m2=1.
    # A1 (2x2) * B1 (2x7) + A2 (2x1) * B2 (1x7)
    # For A1*B1: split p=7 into 2+2+2+1
    #   Use Strassen for each 2x2x2, schoolbook for 2x2x1
    # For A2*B2: schoolbook 2x1x7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(u_vec, v_vec, w_vec):
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

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Part 1: A2 * B2 (2x1 x 7) - schoolbook, 14 terms
    for i in range(n):
        for k in range(p):
            u_vec = [(u_idx(i, 2), 1.0)]
            v_vec = [(v_idx(2, k), 1.0)]
            w_vec = [(w_idx(k, i), 1.0)]
            add_decomp(u_vec, v_vec, w_vec)

    # Part 2: A1 * B1 (2x2 x 7)
    # A1 is A[:, 0:2]. B1 is B[0:2, :].
    # Split B1 into 2+2+2+1 column blocks.
    
    def strassen_decomp(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1), (w_idx(b_col_offset+1, a_row_offset+1), -1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+1), 1), (v_idx(b_row_offset+1, b_col_offset+1), -1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+0), -1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), -1), (w_idx(b_col_offset+1, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+0), -1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1), (u_idx(a_row_offset+1, a_col_offset+1), -1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

    def schoolbook_221(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
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

    # B1 cols 0:2, 2:4, 4:6 use Strassen; col 6 uses schoolbook
    strassen_decomp(0, 0, 0, 0)
    strassen_decomp(0, 0, 0, 2)
    strassen_decomp(0, 0, 0, 4)
    schoolbook_221(0, 0, 0, 6)

    # Convert lists of columns to matrix
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
