import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    Split the 4x5 matrix B into B1 (4x4) and B2 (4x1).
    C = A * B = A * B1 + A * B2 = C1 + C2.
    
    For C1 = A * B1 (4x4 * 4x4), use recursive Strassen (rank 49).
    For C2 = A * B2 (4x4 * 4x1), use naive block decomposition (rank 16).
    
    Total rank = 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i

    def make_A_form(indices):
        form = np.zeros(n*m)
        for i, j in indices:
            form[u_idx(i, j)] = 1
        return form

    def make_B_form(indices):
        form = np.zeros(m*p)
        for j, k in indices:
            form[v_idx(j, k)] = 1
        return form

    # ------------------------------------------------------------------
    # 4x4x4 Strassen Decomposition
    # ------------------------------------------------------------------
    
    # Define the 7 Strassen products for 4x4x4
    # U_i and V_i are 2x2 matrices of linear forms
    
    # M1: U1 = A00 + A11, V1 = B00 + B11
    U1 = [
        [make_A_form([(0,0), (2,2)]), make_A_form([(0,1), (2,3)])],
        [make_A_form([(1,0), (3,2)]), make_A_form([(1,1), (3,3)])]
    ]
    V1 = [
        [make_B_form([(0,0), (2,2)]), make_B_form([(0,1), (2,3)])],
        [make_B_form([(1,0), (3,2)]), make_B_form([(1,1), (3,3)])]
    ]
    
    # M2: U2 = A10 + A11, V2 = B00
    U2 = [
        [make_A_form([(2,0), (0,0)]), make_A_form([(2,1), (0,1)])],
        [make_A_form([(3,0), (1,0)]), make_A_form([(3,1), (1,1)])]
    ]
    V2 = [
        [make_B_form([(0,0)]), make_B_form([(0,1)])],
        [make_B_form([(1,0)]), make_B_form([(1,1)])]
    ]
    
    # M3: U3 = A00, V3 = B01 - B11
    U3 = [
        [make_A_form([(0,0)]), make_A_form([(0,1)])],
        [make_A_form([(1,0)]), make_A_form([(1,1)])]
    ]
    V3 = [
        [make_B_form([(0,1)]), make_B_form([(0,2)])],
        [make_B_form([(1,1)]), make_B_form([(1,2)])]
    ]
    V3[0][0] -= make_B_form([(2,2)])
    V3[0][1] -= make_B_form([(2,3)])
    V3[1][0] -= make_B_form([(3,2)])
    V3[1][1] -= make_B_form([(3,3)])
    
    # M4: U4 = A11, V4 = B10 - B00
    U4 = [
        [make_A_form([(2,2)]), make_A_form([(2,3)])],
        [make_A_form([(3,2)]), make_A_form([(3,3)])]
    ]
    V4 = [
        [make_B_form([(1,0)]), make_B_form([(1,1)])],
        [make_B_form([(2,0)]), make_B_form([(2,1)])]
    ]
    V4[0][0] -= make_B_form([(0,0)])
    V4[0][1] -= make_B_form([(0,1)])
    V4[1][0] -= make_B_form([(1,0)])
    V4[1][1] -= make_B_form([(1,1)])
    
    # M5: U5 = A00 + A01, V5 = B11
    U5 = [
        [make_A_form([(0,0), (0,2)]), make_A_form([(0,1), (0,3)])],
        [make_A_form([(1,0), (1,2)]), make_A_form([(1,1), (1,3)])]
    ]
    V5 = [
        [make_B_form([(2,2)]), make_B_form([(2,3)])],
        [make_B_form([(3,2)]), make_B_form([(3,3)])]
    ]
    
    # M6: U6 = A10 - A00, V6 = B00 + B01
    U6 = [
        [make_A_form([(2,0)]), make_A_form([(2,1)])],
        [make_A_form([(3,0)]), make_A_form([(3,1)])]
    ]
    U6[0][0] -= make_A_form([(0,0)])
    U6[0][1] -= make_A_form([(0,1)])
    U6[1][0] -= make_A_form([(1,0)])
    U6[1][1] -= make_A_form([(1,1)])
    
    V6 = [
        [make_B_form([(0,0), (0,2)]), make_B_form([(0,1), (0,3)])],
        [make_B_form([(1,0), (1,2)]), make_B_form([(1,1), (1,3)])]
    ]
    
    # M7: U7 = A01 - A11, V7 = B10 + B11
    U7 = [
        [make_A_form([(0,2)]), make_A_form([(0,3)])],
        [make_A_form([(1,2)]), make_A_form([(1,3)])]
    ]
    U7[0][0] -= make_A_form([(2,2)])
    U7[0][1] -= make_A_form([(2,3)])
    U7[1][0] -= make_A_form([(3,2)])
    U7[1][1] -= make_A_form([(3,3)])
    
    V7 = [
        [make_B_form([(1,0), (2,2)]), make_B_form([(1,1), (2,3)])],
        [make_B_form([(2,0), (3,2)]), make_B_form([(2,1), (3,3)])]
    ]
    
    # Helper to add the 7 terms of a 2x2x2 Strassen product
    # U, V are 2x2 lists of forms
    # placements: list of (row_start, col_start) for the 2x2 output block
    def add_strassen_2x2_terms(U, V, placements):
        # P1 = (U00+U11)(V00+V11) -> L00(+1), L11(+1)
        u_p1 = U[0][0] + U[1][1]
        v_p1 = V[0][0] + V[1][1]
        w_p1 = np.zeros(p*n)
        for rs, cs in placements:
            w_p1[w_idx(cs, rs)] = 1       # L00 -> C[rs, cs]
            w_p1[w_idx(cs+1, rs+1)] = 1   # L11 -> C[rs+1, cs+1]
        add_term(u_p1, v_p1, w_p1)
        
        # P2 = (U10+U11)V00 -> L10(+1), L11(-1)
        u_p2 = U[1][0] + U[1][1]
        v_p2 = V[0][0]
        w_p2 = np.zeros(p*n)
        for rs, cs in placements:
            w_p2[w_idx(cs, rs+1)] = 1     # L10 -> C[rs+1, cs]
            w_p2[w_idx(cs+1, rs+1)] = -1  # L11 -> C[rs+1, cs+1]
        add_term(u_p2, v_p2, w_p2)
        
        # P3 = U00(V01-V11) -> L01(+1), L11(+1)
        u_p3 = U[0][0]
        v_p3 = V[0][1] - V[1][1]
        w_p3 = np.zeros(p*n)
        for rs, cs in placements:
            w_p3[w_idx(cs+1, rs)] = 1     # L01 -> C[rs, cs+1]
            w_p3[w_idx(cs+1, rs+1)] = 1   # L11 -> C[rs+1, cs+1]
        add_term(u_p3, v_p3, w_p3)
        
        # P4 = U11(V10-V00) -> L00(+1), L10(+1)
        u_p4 = U[1][1]
        v_p4 = V[1][0] - V[0][0]
        w_p4 = np.zeros(p*n)
        for rs, cs in placements:
            w_p4[w_idx(cs, rs)] = 1       # L00 -> C[rs, cs]
            w_p4[w_idx(cs, rs+1)] = 1     # L10 -> C[rs+1, cs]
        add_term(u_p4, v_p4, w_p4)
        
        # P5 = (U00+U01)V11 -> L00(-1), L01(+1)
        u_p5 = U[0][0] + U[0][1]
        v_p5 = V[1][1]
        w_p5 = np.zeros(p*n)
        for rs, cs in placements:
            w_p5[w_idx(cs, rs)] = -1      # L00 -> C[rs, cs]
            w_p5[w_idx(cs+1, rs)] = 1     # L01 -> C[rs, cs+1]
        add_term(u_p5, v_p5, w_p5)
        
        # P6 = (U10-U00)(V00+V01) -> L11(+1)
        u_p6 = U[1][0] - U[0][0]
        v_p6 = V[0][0] + V[0][1]
        w_p6 = np.zeros(p*n)
        for rs, cs in placements:
            w_p6[w_idx(cs+1, rs+1)] = 1   # L11 -> C[rs+1, cs+1]
        add_term(u_p6, v_p6, w_p6)
        
        # P7 = (U01-U11)(V10+V11) -> L00(+1)
        u_p7 = U[0][1] - U[1][1]
        v_p7 = V[1][0] + V[1][1]
        w_p7 = np.zeros(p*n)
        for rs, cs in placements:
            w_p7[w_idx(cs, rs)] = 1       # L00 -> C[rs, cs]
        add_term(u_p7, v_p7, w_p7)

    # Apply Strassen to the 7 main products
    # M1 -> C00 (0,0) and C11 (2,2)
    add_strassen_2x2_terms(U1, V1, [(0,0), (2,2)])
    
    # M2 -> C10 (2,0) and C11 (2,2)
    add_strassen_2x2_terms(U2, V2, [(2,0), (2,2)])
    
    # M3 -> C01 (0,2) and C11 (2,2)
    add_strassen_2x2_terms(U3, V3, [(0,2), (2,2)])
    
    # M4 -> C00 (0,0) and C10 (2,0)
    add_strassen_2x2_terms(U4, V4, [(0,0), (2,0)])
    
    # M5 -> C00 (0,0) and C01 (0,2)
    add_strassen_2x2_terms(U5, V5, [(0,0), (0,2)])
    
    # M6 -> C11 (2,2)
    add_strassen_2x2_terms(U6, V6, [(2,2)])
    
    # M7 -> C00 (0,0)
    add_strassen_2x2_terms(U7, V7, [(0,0)])
    
    # ------------------------------------------------------------------
    # 4x4x1 Decomposition (C2)
    # ------------------------------------------------------------------
    # C2 = A * B2, where B2 is col 4 of B.
    # C2 rows 0-1: A00*b0 + A01*b1
    # C2 rows 2-3: A10*b0 + A11*b1
    
    def add_naive_2x2x1(ai, aj, bj, bk):
        # A block at ai,aj (2x2), B col at bj,bk (2x1)
        # C rows ai, ai+1, col bk
        
        # a=A[ai,aj], b=A[ai,aj+1], c=A[ai+1,aj], d=A[ai+1,aj+1]
        # e=B[bj,bk], g=B[bj+1,bk]
        
        # Term 1: a*e -> C[ai,bk]
        u1 = np.zeros(n*m); u1[u_idx(ai,aj)] = 1
        v1 = np.zeros(m*p); v1[v_idx(bj,bk)] = 1
        w1 = np.zeros(p*n); w1[w_idx(bk,ai)] = 1
        add_term(u1, v1, w1)
        
        # Term 2: b*g -> C[ai,bk]
        u2 = np.zeros(n*m); u2[u_idx(ai,aj+1)] = 1
        v2 = np.zeros(m*p); v2[v_idx(bj+1,bk)] = 1
        w2 = np.zeros(p*n); w2[w_idx(bk,ai)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: c*e -> C[ai+1,bk]
        u3 = np.zeros(n*m); u3[u_idx(ai+1,aj)] = 1
        v3 = np.zeros(m*p); v3[v_idx(bj,bk)] = 1
        w3 = np.zeros(p*n); w3[w_idx(bk,ai+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: d*g -> C[ai+1,bk]
        u4 = np.zeros(n*m); u4[u_idx(ai+1,aj+1)] = 1
        v4 = np.zeros(m*p); v4[v_idx(bj+1,bk)] = 1
        w4 = np.zeros(p*n); w4[w_idx(bk,ai+1)] = 1
        add_term(u4, v4, w4)

    # Top 2 rows of C2: A00*b0 + A01*b1
    # A00 is at (0,0), b0 is B[0:2, 4] -> bj=0, bk=4
    add_naive_2x2x1(0, 0, 0, 4)
    # A01 is at (0,2), b1 is B[2:4, 4] -> bj=2, bk=4
    add_naive_2x2x1(0, 2, 2, 4)
    
    # Bottom 2 rows of C2: A10*b0 + A11*b1
    # A10 is at (2,0), b0 is B[0:2, 4] -> bj=0, bk=4
    add_naive_2x2x1(2, 0, 0, 4)
    # A11 is at (2,2), b1 is B[2:4, 4] -> bj=2, bk=4
    add_naive_2x2x1(2, 2, 2, 4)
    
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    return {"U": U, "V": V, "W": W}
