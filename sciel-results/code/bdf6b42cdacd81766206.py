"""Program for matrix multiplication 2x5 by 5x6 with rank-minimal decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Partition A (2x5) into A1 (2x2), A2 (2x2), A3 (2x1)
    # Partition B (5x6) into 3 rows of 2x2 blocks: B11,B12,B13 / B21,B22,B23 / B31,B32,B33
    #
    # Strassen for 2x2x2 (rank 7):
    #   M1 = (a00+a11)(b00+b11)
    #   M2 = (a10+a11)b00
    #   M3 = a00(b01-b11)
    #   M4 = a11(b10-b00)
    #   M5 = (a00+a01)b11
    #   M6 = (a10-a00)(b00+b01)
    #   M7 = (a01-a11)(b10+b11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # Schoolbook for 2x1x2 (rank 4):
    #   C00 = a0*b0, C01 = a0*b1, C10 = a1*b0, C11 = a1*b1
    
    U = []  # will be list of lists, shape (10, rank)
    V = []  # shape (30, rank)
    W = []  # shape (12, rank)
    
    rank = 0
    
    def add_strassen(row_offset_a, col_offset_a, col_offset_b):
        """Add Strassen decomposition for A[row_offset_a:row_offset_a+2, col_offset_a:col_offset_a+2] * B[...2x2 block at col_offset_b]"""
        nonlocal rank
        # A rows: row_offset_a, row_offset_a+1 (0-indexed in A's 2 rows)
        # A cols: col_offset_a, col_offset_a+1 (in A's 5 cols)
        # B rows: corresponding to A cols, so B row indices are col_offset_a, col_offset_a+1
        # B cols: col_offset_b, col_offset_b+1 (in B's 6 cols)
        
        # For each of 7 Strassen terms, add one column
        # M1 = (a00+a11)(b00+b11)
        # U row for A: rows row_offset_a and row_offset_a+1, cols col_offset_a and col_offset_a+1
        # V row for B: rows col_offset_a and col_offset_a+1, cols col_offset_b and col_offset_b+1
        # W: contributes to output rows
        
        # Let's define local indices:
        # a00 -> A[row_offset_a, col_offset_a] -> U index = row_offset_a * 5 + col_offset_a
        # a11 -> A[row_offset_a+1, col_offset_a+1]
        # b00 -> B[col_offset_a, col_offset_b] -> V index = col_offset_a * 6 + col_offset_b
        # b11 -> B[col_offset_a+1, col_offset_b+1]
        
        # Output C is 2x2: C00, C01, C10, C11
        # C00 -> W index = 0*2 + 0 = 0 (output row 0, col 0) -- wait, W convention: W[k*n+i, r]
        # n=2, p=6. W has p*n = 12 rows. Index k*n + i where k is B's col index, i is A's row index.
        # So output element (i, k) where i in {0,1} (A row) and k in {0,1} (B col within this block)
        # But B col is col_offset_b + k, so global B col = col_offset_b + k
        # W index = (col_offset_b + k) * 2 + i
        
        # For Strassen on A[ra:ra+2, ca:ca+2] * B[ca:ca+2, cb:cb+2]:
        # Local A row indices: ra, ra+1
        # Local A col indices: ca, ca+1
        # Local B row indices: ca, ca+1 (these are the inner dimension)
        # Local B col indices: cb, cb+1
        #
        # Output local (i, j) where i in {0,1}, j in {0,1}
        # Global A row: ra + i
        # Global B col: cb + j
        # W index = (cb + j) * n + (ra + i) = (cb + j) * 2 + ra + i
        
        for s in range(7):
            col = rank
            rank += 1
            
            # Determine U coefficients (for A rows ra, ra+1; cols ca, ca+1)
            u_vals = [0.0] * (2 * 5)  # 10 entries for A
            # V coefficients (for B rows ca, ca+1; cols cb, cb+1)
            v_vals = [0.0] * (2 * 6)  # 12 entries for B
            # W coefficients (for output positions)
            w_vals = [0.0] * (2 * 6)  # 12 entries for C
            
            a00_idx = ra * 5 + ca
            a01_idx = ra * 5 + ca + 1
            a10_idx = (ra + 1) * 5 + ca
            a11_idx = (ra + 1) * 5 + ca + 1
            
            b00_idx = ca * 6 + cb
            b01_idx = ca * 6 + cb + 1
            b10_idx = (ca + 1) * 6 + cb
            b11_idx = (ca + 1) * 6 + cb + 1
            
            c00_idx = cb * 2 + ra
            c01_idx = cb * 2 + ra + 1
            c10_idx = (cb + 1) * 2 + ra
            c11_idx = (cb + 1) * 2 + ra + 1
            
            if s == 0:  # M1 = (a00+a11)(b00+b11)
                u_vals[a00_idx] = 1
                u_vals[a11_idx] = 1
                v_vals[b00_idx] = 1
                v_vals[b11_idx] = 1
                w_vals[c00_idx] = 1
                w_vals[c11_idx] = 1
            elif s == 1:  # M2 = (a10+a11)b00
                u_vals[a10_idx] = 1
                u_vals[a11_idx] = 1
                v_vals[b00_idx] = 1
                w_vals[c10_idx] = 1
                w_vals[c11_idx] = -1
            elif s == 2:  # M3 = a00(b01-b11)
                u_vals[a00_idx] = 1
                v_vals[b01_idx] = 1
                v_vals[b11_idx] = -1
                w_vals[c01_idx] = 1
                w_vals[c11_idx] = 1
            elif s == 3:  # M4 = a11(b10-b00)
                u_vals[a11_idx] = 1
                v_vals[b10_idx] = 1
                v_vals[b00_idx] = -1
                w_vals[c00_idx] = 1
                w_vals[c10_idx] = 1
            elif s == 4:  # M5 = (a00+a01)b11
                u_vals[a00_idx] = 1
                u_vals[a01_idx] = 1
                v_vals[b11_idx] = 1
                w_vals[c00_idx] = -1
                w_vals[c01_idx] = 1
            elif s == 5:  # M6 = (a10-a00)(b00+b01)
                u_vals[a10_idx] = 1
                u_vals[a00_idx] = -1
                v_vals[b00_idx] = 1
                v_vals[b01_idx] = 1
                w_vals[c11_idx] = 1
            elif s == 6:  # M7 = (a01-a11)(b10+b11)
                u_vals[a01_idx] = 1
                u_vals[a11_idx] = -1
                v_vals[b10_idx] = 1
                v_vals[b11_idx] = 1
                w_vals[c00_idx] = 1
            
            U.append([0.0] * 0)  # placeholder, we'll build later
            # Actually let's build U, V, W as we go
            # U has 10 rows, V has 30 rows, W has 12 rows
            # For now, store column data
            
            # We'll accumulate
            if not hasattr(add_strassen, 'columns'):
                add_strassen.columns = []
            add_strassen.columns.append((u_vals, v_vals, w_vals))
    
    def add_schoolbook_2x1x2(row_offset_a, col_offset_a, col_offset_b):
        """Add schoolbook for A[ra:ra+2, ca:ca+1] * B[ca:ca+1, cb:cb+2]"""
        nonlocal rank
        ra = row_offset_a
        ca = col_offset_a
        cb = col_offset_b
        
        a0_idx = ra * 5 + ca
        a1_idx = (ra + 1) * 5 + ca
        b0_idx = ca * 6 + cb
        b1_idx = ca * 6 + cb + 1
        
        c00_idx = cb * 2 + ra
        c01_idx = cb * 2 + ra + 1
        c10_idx = (cb + 1) * 2 + ra
        c11_idx = (cb + 1) * 2 + ra + 1
        
        for s in range(4):
            col = rank
            rank += 1
            
            u_vals = [0.0] * 10
            v_vals = [0.0] * 30
            w_vals = [0.0] * 12
            
            if s == 0:  # a0 * b0
                u_vals[a0_idx] = 1
                v_vals[b0_idx] = 1
                w_vals[c00_idx] = 1
            elif s == 1:  # a0 * b1
                u_vals[a0_idx] = 1
                v_vals[b1_idx] = 1
                w_vals[c01_idx] = 1
            elif s == 2:  # a1 * b0
                u_vals[a1_idx] = 1
                v_vals[b0_idx] = 1
                w_vals[c10_idx] = 1
            elif s == 3:  # a1 * b1
                u_vals[a1_idx] = 1
                v_vals[b1_idx] = 1
                w_vals[c11_idx] = 1
            
            if not hasattr(add_schoolbook_2x1x2, 'columns'):
                add_schoolbook_2x1x2.columns = []
            add_schoolbook_2x1x2.columns.append((u_vals, v_vals, w_vals))
    
    # A is 2x5: A1 = A[:, 0:2], A2 = A[:, 2:4], A3 = A[:, 4:5]
    # B is 5x6: 
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:6]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:6]
    #   B31 = B[4:5, 0:2], B32 = B[4:5, 2:4], B33 = B[4:5, 4:6]
    
    # A1 * B11: ra=0, ca=0, cb=0
    add_strassen(0, 0, 0)
    # A1 * B12: ra=0, ca=0, cb=2
    add_strassen(0, 0, 2)
    # A1 * B13: ra=0, ca=0, cb=4
    add_strassen(0, 0, 4)
    
    # A2 * B21: ra=0, ca=2, cb=0
    add_strassen(0, 2, 0)
    # A2 * B22: ra=0, ca=2, cb=2
    add_strassen(0, 2, 2)
    # A2 * B23: ra=0, ca=2, cb=4
    add_strassen(0, 2, 4)
    
    # A3 * B31: ra=0, ca=4, cb=0
    add_schoolbook_2x1x2(0, 4, 0)
    # A3 * B32: ra=0, ca=4, cb=2
    add_schoolbook_2x1x2(0, 4, 2)
    # A3 * B33: ra=0, ca=4, cb=4
    add_schoolbook_2x1x2(0, 4, 4)
    
    # Now assemble U, V, W from all columns
    U_cols = getattr(add_strassen, 'columns', [])
    V_cols = getattr(add_schoolbook_2x1x2, 'columns', [])
    
    # Wait, I need to combine both. Let me redo this more carefully.
    # Actually, let me just collect all columns in a single list.
    
    # Let me restart with a cleaner approach
    
    U_list = []
    V_list = []
    W_list = []
    
    def strassen_cols(ra, ca, cb):
        """Return list of (u, v, w) tuples for Strassen on A[ra:ra+2, ca:ca+2] * B[ca:ca+2, cb:cb+2]"""
        cols = []
        a00_idx = ra * 5 + ca
        a01_idx = ra * 5 + ca + 1
        a10_idx = (ra + 1) * 5 + ca
        a11_idx = (ra + 1) * 5 + ca + 1
        
        b00_idx = ca * 6 + cb
        b01_idx = ca * 6 + cb + 1
        b10_idx = (ca + 1) * 6 + cb
        b11_idx = (ca + 1) * 6 + cb + 1
        
        c00_idx = cb * 2 + ra
        c01_idx = cb * 2 + ra + 1
        c10_idx = (cb + 1) * 2 + ra
        c11_idx = (cb + 1) * 2 + ra + 1
        
        for s in range(7):
            u = [0.0] * 10
            v = [0.0] * 30
            w = [0.0] * 12
            if s == 0:
                u[a00_idx] = 1; u[a11_idx] = 1
                v[b00_idx] = 1; v[b11_idx] = 1
                w[c00_idx] = 1; w[c11_idx] = 1
            elif s == 1:
                u[a10_idx] = 1; u[a11_idx] = 1
                v[b00_idx] = 1
                w[c10_idx] = 1; w[c11_idx] = -1
            elif s == 2:
                u[a00_idx] = 1
                v[b01_idx] = 1; v[b11_idx] = -1
                w[c01_idx] = 1; w[c11_idx] = 1
            elif s == 3:
                u[a11_idx] = 1
                v[b10_idx] = 1; v[b00_idx] = -1
                w[c00_idx] = 1; w[c10_idx] = 1
            elif s == 4:
                u[a00_idx] = 1; u[a01_idx] = 1
                v[b11_idx] = 1
                w[c00_idx] = -1; w[c01_idx] = 1
            elif s == 5:
                u[a10_idx] = 1; u[a00_idx] = -1
                v[b00_idx] = 1; v[b01_idx] = 1
                w[c11_idx] = 1
            elif s == 6:
                u[a01_idx] = 1; u[a11_idx] = -1
                v[b10_idx] = 1; v[b11_idx] = 1
                w[c00_idx] = 1
            cols.append((u, v, w))
        return cols
    
    def schoolbook_2x1x2_cols(ra, ca, cb):
        """Return list of (u, v, w) tuples for schoolbook on A[ra:ra+2, ca:ca+1] * B[ca:ca+1, cb:cb+2]"""
        cols = []
        a0_idx = ra * 5 + ca
        a1_idx = (ra + 1) * 5 + ca
        b0_idx = ca * 6 + cb
        b1_idx = ca * 6 + cb + 1
        
        c00_idx = cb * 2 + ra
        c01_idx = cb * 2 + ra + 1
        c10_idx = (cb + 1) * 2 + ra
        c11_idx = (cb + 1) * 2 + ra + 1
        
        for s in range(4):
            u = [0.0] * 10
            v = [0.0] * 30
            w = [0.0] * 12
            if s == 0:
                u[a0_idx] = 1
                v[b0_idx] = 1
                w[c00_idx] = 1
            elif s == 1:
                u[a0_idx] = 1
                v[b1_idx] = 1
                w[c01_idx] = 1
            elif s == 2:
                u[a1_idx] = 1
                v[b0_idx] = 1
                w[c10_idx] = 1
            elif s == 3:
                u[a1_idx] = 1
                v[b1_idx] = 1
                w[c11_idx] = 1
            cols.append((u, v, w))
        return cols
    
    all_cols = []
    
    # A1 (rows 0-1, cols 0-1 of A) * B11 (rows 0-1, cols 0-1 of B)
    all_cols.extend(strassen_cols(0, 0, 0))
    # A1 * B12
    all_cols.extend(strassen_cols(0, 0, 2))
    # A1 * B13
    all_cols.extend(strassen_cols(0, 0, 4))
    
    # A2 (rows 0-1, cols 2-3 of A) * B21 (rows 2-3, cols 0-1 of B)
    all_cols.extend(strassen_cols(0, 2, 0))
    # A2 * B22
    all_cols.extend(strassen_cols(0, 2, 2))
    # A2 * B23
    all_cols.extend(strassen_cols(0, 2, 4))
    
    # A3 (rows 0-1, col 4 of A) * B31 (row 4, cols 0-1 of B)
    all_cols.extend(schoolbook_2x1x2_cols(0, 4, 0))
    # A3 * B32
    all_cols.extend(schoolbook_2x1x2_cols(0, 4, 2))
    # A3 * B33
    all_cols.extend(schoolbook_2x1x2_cols(0, 4, 4))
    
    r = len(all_cols)
    U_mat = [[0.0] * r for _ in range(10)]
    V_mat = [[0.0] * r for _ in range(30)]
    W_mat = [[0.0] * r for _ in range(12)]
    
    for c, (u, v, w) in enumerate(all_cols):
        for i in range(10):
            U_mat[i][c] = u[i]
        for i in range(30):
            V_mat[i][c] = v[i]
        for i in range(12):
            W_mat[i][c] = w[i]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
