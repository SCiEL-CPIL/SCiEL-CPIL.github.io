import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 52
    # 28 from 2x4x4 part
    # 24 from 2x4x3 part
    
    U = np.zeros((n*m, 52), dtype=float)
    V = np.zeros((m*p, 52), dtype=float)
    W = np.zeros((p*n, 52), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Strassen 2x2x2 terms
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = XY
    # Indices for X: (i, j) -> i*2+j
    # Indices for Y: (j, k) -> j*2+k
    # Indices for Z: (k, i) -> k*2+i (following W[k*n+i] convention, here n=2)
    
    # We need to map local 2x2x2 to global indices.
    
    # Part 1: 2x4x4 (Rank 28)
    # A (2x4) split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # B (4x4) split into B_T (2x4, rows 0,1) and B_B (2x4, rows 2,3)
    # C (2x4) is sum of A_L B_T and A_R B_B
    
    # A_L B_T is 2x2x4
    # B_T is 2x4, split into Y1 (2x2, cols 0,1) and Y2 (2x2, cols 2,3)
    # A_L Y1 is 2x2x2
    # A_L Y2 is 2x2x2
    
    # A_R B_B is 2x2x4
    # B_B is 2x4, split into Y3 (2x2, cols 0,1) and Y4 (2x2, cols 2,3)
    # A_R Y3 is 2x2x2
    # A_R Y4 is 2x2x2
    
    # We need a function to add a 2x2x2 Strassen block
    # Given offsets for A (row_off, col_off), B (row_off, col_off), C (row_off, col_off)
    # Note: B is m x p. In 2x2x2, B is 2x2.
    # Global A index: i*m + j. Local i in 0..1, j in 0..1.
    # Global B index: j*p + k. Local j in 0..1, k in 0..1.
    # Global C index: k*n + i. Local k in 0..1, i in 0..1.
    
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        # a_row_off: offset for i in A
        # a_col_off: offset for j in A
        # b_row_off: offset for j in B
        # b_col_off: offset for k in B
        # c_row_off: offset for k in C
        # c_col_off: offset for i in C
        
        # Strassen multipliers
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y01-y11)
        # M4 = x11(y10-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # Z00 = M1 - M4 + M5 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We add 7 terms.
        # Term 1: M1
        # U: x00, x11
        # V: y00, y11
        # W: z00, z11
        
        # Indices in local 2x2
        # x00: 0, x01: 1, x10: 2, x11: 3
        # y00: 0, y01: 1, y10: 2, y11: 3
        # z00: 0 (k=0, i=0), z01: 1 (k=0, i=1) -> Wait.
        # W index is k*n + i.
        # Local n=2.
        # z00 (i=0, k=0) -> 0*2+0 = 0.
        # z01 (i=1, k=0) -> 0*2+1 = 1.
        # z10 (i=0, k=1) -> 1*2+0 = 2.
        # z11 (i=1, k=1) -> 1*2+1 = 3.
        
        # Global indices
        # A: a_row_off + i, a_col_off + j -> idx = (a_row_off+i)*m + (a_col_off+j)
        # B: b_row_off + j, b_col_off + k -> idx = (b_row_off+j)*p + (b_col_off+k)
        # W: c_row_off + k, c_col_off + i -> idx = (c_row_off+k)*n + (c_col_off+i)
        
        # Term 1: M1
        # U: x00 (1), x11 (1)
        # V: y00 (1), y11 (1)
        # W: z00 (1), z11 (1)
        nonlocal col
        U[(a_row_off+0)*m + (a_col_off+0), col] = 1
        U[(a_row_off+1)*m + (a_col_off+1), col] = 1
        V[(b_row_off+0)*p + (b_col_off+0), col] = 1
        V[(b_row_off+1)*p + (b_col_off+1), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = 1
        W[(c_row_off+1)*n + (c_col_off+1), col] = 1
        col += 1
        
        # Term 2: M2
        # U: x10 (1), x11 (1)
        # V: y00 (1)
        # W: z10 (1)
        U[(a_row_off+1)*m + (a_col_off+0), col] = 1
        U[(a_row_off+1)*m + (a_col_off+1), col] = 1
        V[(b_row_off+0)*p + (b_col_off+0), col] = 1
        W[(c_row_off+1)*n + (c_col_off+0), col] = 1
        col += 1
        
        # Term 3: M3
        # U: x00 (1)
        # V: y01 (1), y11 (-1)
        # W: z01 (1)
        U[(a_row_off+0)*m + (a_col_off+0), col] = 1
        V[(b_row_off+0)*p + (b_col_off+1), col] = 1
        V[(b_row_off+1)*p + (b_col_off+1), col] = -1
        W[(c_row_off+0)*n + (c_col_off+1), col] = 1
        col += 1
        
        # Term 4: M4
        # U: x11 (1)
        # V: y10 (1), y00 (-1)
        # W: z10 (1), z00 (-1)
        U[(a_row_off+1)*m + (a_col_off+1), col] = 1
        V[(b_row_off+1)*p + (b_col_off+0), col] = 1
        V[(b_row_off+0)*p + (b_col_off+0), col] = -1
        W[(c_row_off+1)*n + (c_col_off+0), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = -1
        col += 1
        
        # Term 5: M5
        # U: x00 (1), x01 (1)
        # V: y11 (1)
        # W: z00 (1), z01 (1)
        U[(a_row_off+0)*m + (a_col_off+0), col] = 1
        U[(a_row_off+0)*m + (a_col_off+1), col] = 1
        V[(b_row_off+1)*p + (b_col_off+1), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = 1
        W[(c_row_off+0)*n + (c_col_off+1), col] = 1
        col += 1
        
        # Term 6: M6
        # U: x10 (1), x00 (-1)
        # V: y00 (1), y01 (1)
        # W: z11 (1)
        U[(a_row_off+1)*m + (a_col_off+0), col] = 1
        U[(a_row_off+0)*m + (a_col_off+0), col] = -1
        V[(b_row_off+0)*p + (b_col_off+0), col] = 1
        V[(b_row_off+0)*p + (b_col_off+1), col] = 1
        W[(c_row_off+1)*n + (c_col_off+1), col] = 1
        col += 1
        
        # Term 7: M7
        # U: x01 (1), x11 (-1)
        # V: y10 (1), y11 (1)
        # W: z00 (-1), z11 (1)
        U[(a_row_off+0)*m + (a_col_off+1), col] = 1
        U[(a_row_off+1)*m + (a_col_off+1), col] = -1
        V[(b_row_off+1)*p + (b_col_off+0), col] = 1
        V[(b_row_off+1)*p + (b_col_off+1), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = -1
        W[(c_row_off+1)*n + (c_col_off+1), col] = 1
        col += 1

    # Part 1: 2x4x4
    # A_L (rows 0-1, cols 0-1 of A)
    # B_T (rows 0-1 of B, cols 0-3 of B)
    # C (rows 0-3 of C, cols 0-1 of C)
    
    # A_L Y1 (Y1 is B_T cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A_L Y2 (Y2 is B_T cols 2-3)
    add_strassen_block(0, 0, 0, 2, 2, 0)
    
    # A_R (rows 0-1, cols 2-3 of A)
    # B_B (rows 2-3 of B, cols 0-3 of B)
    # C (rows 0-3 of C, cols 0-1 of C)
    
    # A_R Y3 (Y3 is B_B cols 0-1)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # A_R Y4 (Y4 is B_B cols 2-3)
    add_strassen_block(0, 2, 2, 2, 2, 0)
    
    # Part 2: 2x4x3 (Rank 24)
    # A_L, A_R same as before.
    # B_T3 (rows 0-1 of B, cols 4-6 of B) -> 2x3
    # B_B3 (rows 2-3 of B, cols 4-6 of B) -> 2x3
    # C3 (rows 4-6 of C, cols 0-1 of C)
    
    # We use 2x2x3 decomposition (Rank 12) for A_L B_T3 and A_R B_B3.
    # 2x2x3 is sum of three 2x2x1.
    
    def add_2x2x1_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        # X (2x2), Y (2x1)
        # Z0 = x00 y0 + x01 y1
        # Z1 = x10 y0 + x11 y1
        nonlocal col
        # Term 1: x00 y0 -> Z0
        U[(a_row_off+0)*m + (a_col_off+0), col] = 1
        V[(b_row_off+0)*p + (b_col_off+0), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = 1
        col += 1
        
        # Term 2: x01 y1 -> Z0
        U[(a_row_off+0)*m + (a_col_off+1), col] = 1
        V[(b_row_off+1)*p + (b_col_off+0), col] = 1
        W[(c_row_off+0)*n + (c_col_off+0), col] = 1
        col += 1
        
        # Term 3: x10 y0 -> Z1
        U[(a_row_off+1)*m + (a_col_off+0), col] = 1
        V[(b_row_off+0)*p + (b_col_off+0), col] = 1
        W[(c_row_off+1)*n + (c_col_off+0), col] = 1
        col += 1
        
        # Term 4: x11 y1 -> Z1
        U[(a_row_off+1)*m + (a_col_off+1), col] = 1
        V[(b_row_off+1)*p + (b_col_off+0), col] = 1
        W[(c_row_off+1)*n + (c_col_off+0), col] = 1
        col += 1

    # A_L B_T3
    # B_T3 is 2x3. Split into 3 columns.
    # Col 4: b_col_off=4
    add_2x2x1_block(0, 0, 0, 4, 4, 0)
    # Col 5: b_col_off=5
    add_2x2x1_block(0, 0, 0, 5, 5, 0)
    # Col 6: b_col_off=6
    add_2x2x1_block(0, 0, 0, 6, 6, 0)
    
    # A_R B_B3
    # Col 4: b_col_off=4
    add_2x2x1_block(0, 2, 2, 4, 4, 0)
    # Col 5: b_col_off=5
    add_2x2x1_block(0, 2, 2, 5, 5, 0)
    # Col 6: b_col_off=6
    add_2x2x1_block(0, 2, 2, 6, 6, 0)
    
    return {"U": U, "V": V, "W": W}

