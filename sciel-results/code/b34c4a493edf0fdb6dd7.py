import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 28
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients
    # M1: (x00+x11)(y00+y11) -> z00+1, z11+1
    # M2: (x10+x11)y00 -> z10+1, z11-1
    # M3: x00(y01-y11) -> z01+1, z11+1
    # M4: x11(y10-y00) -> z00+1, z10+1
    # M5: (x00+x01)y11 -> z00-1, z01+1
    # M6: (x10-x00)(y00+y01) -> z11+1
    # M7: (x01-x11)(y10+y11) -> z00+1
    
    # Block 1: A1 (0:2, 0:2) * B11 (0:2, 0:2) -> C1 (0:2, 0:2)
    # A1 indices: A[0,0], A[0,1], A[1,0], A[1,1]
    # B11 indices: B[0,0], B[0,1], B[1,0], B[1,1]
    # C1 indices: C[0,0], C[0,1], C[1,0], C[1,1]
    
    def add_strassen_term(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, term_id, col_idx):
        # term_id 0..6
        # a_rows, a_cols: indices of X in A
        # b_rows, b_cols: indices of Y in B
        # c_rows, c_cols: indices of Z in C
        
        # Define U, V, W for this term
        u_vals = {}
        v_vals = {}
        w_vals = {}
        
        # M1
        if term_id == 0:
            u_vals[(a_rows[0], a_cols[0])] = 1
            u_vals[(a_rows[1], a_cols[1])] = 1
            v_vals[(b_rows[0], b_cols[0])] = 1
            v_vals[(b_rows[1], b_cols[1])] = 1
            w_vals[(c_rows[0], c_cols[0])] = 1
            w_vals[(c_rows[1], c_cols[1])] = 1
        # M2
        elif term_id == 1:
            u_vals[(a_rows[1], a_cols[0])] = 1
            u_vals[(a_rows[1], a_cols[1])] = 1
            v_vals[(b_rows[0], b_cols[0])] = 1
            w_vals[(c_rows[1], c_cols[0])] = 1
            w_vals[(c_rows[1], c_cols[1])] = -1
        # M3
        elif term_id == 2:
            u_vals[(a_rows[0], a_cols[0])] = 1
            v_vals[(b_rows[0], b_cols[1])] = 1
            v_vals[(b_rows[1], b_cols[1])] = -1
            w_vals[(c_rows[0], c_cols[1])] = 1
            w_vals[(c_rows[1], c_cols[1])] = 1
        # M4
        elif term_id == 3:
            u_vals[(a_rows[1], a_cols[1])] = 1
            v_vals[(b_rows[1], b_cols[0])] = 1
            v_vals[(b_rows[0], b_cols[0])] = -1
            w_vals[(c_rows[0], c_cols[0])] = 1
            w_vals[(c_rows[1], c_cols[0])] = 1
        # M5
        elif term_id == 4:
            u_vals[(a_rows[0], a_cols[0])] = 1
            u_vals[(a_rows[0], a_cols[1])] = 1
            v_vals[(b_rows[1], b_cols[1])] = 1
            w_vals[(c_rows[0], c_cols[0])] = -1
            w_vals[(c_rows[0], c_cols[1])] = 1
        # M6
        elif term_id == 5:
            u_vals[(a_rows[1], a_cols[0])] = 1
            u_vals[(a_rows[0], a_cols[0])] = -1
            v_vals[(b_rows[0], b_cols[0])] = 1
            v_vals[(b_rows[0], b_cols[1])] = 1
            w_vals[(c_rows[1], c_cols[1])] = 1
        # M7
        elif term_id == 6:
            u_vals[(a_rows[0], a_cols[1])] = 1
            u_vals[(a_rows[1], a_cols[1])] = -1
            v_vals[(b_rows[1], b_cols[0])] = 1
            v_vals[(b_rows[1], b_cols[1])] = 1
            w_vals[(c_rows[0], c_cols[0])] = 1

        for (r, c), val in u_vals.items():
            U[r * m + c, col_idx] = val
        for (r, c), val in v_vals.items():
            V[r * p + c, col_idx] = val
        for (r, c), val in w_vals.items():
            W[c * n + r, col_idx] = val # Note: W index is k*n+i. c is k, r is i.

    # A1: rows 0-1, cols 0-1
    # B11: rows 0-1, cols 0-1
    # C1: rows 0-1, cols 0-1
    a1_r = [0, 1]
    a1_c = [0, 1]
    b11_r = [0, 1]
    b11_c = [0, 1]
    c1_r = [0, 1]
    c1_c = [0, 1]
    
    for t in range(7):
        add_strassen_term(a1_r, a1_c, b11_r, b11_c, c1_r, c1_c, t, col)
        col += 1
        
    # A2: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # C1: rows 0-1, cols 0-1
    a2_r = [0, 1]
    a2_c = [2, 3]
    b21_r = [2, 3]
    b21_c = [0, 1]
    
    for t in range(7):
        add_strassen_term(a2_r, a2_c, b21_r, b21_c, c1_r, c1_c, t, col)
        col += 1
        
    # A1: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # C2: rows 0-1, cols 2-3
    b12_r = [0, 1]
    b12_c = [2, 3]
    c2_r = [0, 1]
    c2_c = [2, 3]
    
    for t in range(7):
        add_strassen_term(a1_r, a1_c, b12_r, b12_c, c2_r, c2_c, t, col)
        col += 1
        
    # A2: rows 0-1, cols 2-3
    # B22: rows 2-3, cols 2-3
    # C2: rows 0-1, cols 2-3
    b22_r = [2, 3]
    b22_c = [2, 3]
    
    for t in range(7):
        add_strassen_term(a2_r, a2_c, b22_r, b22_c, c2_r, c2_c, t, col)
        col += 1
        
    return {"U": U, "V": V, "W": W}
