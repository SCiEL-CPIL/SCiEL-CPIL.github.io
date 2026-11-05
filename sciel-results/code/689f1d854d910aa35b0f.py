def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen block addition
    def add_strassen(a_rows, a_cols, b_rows, b_cols, start_col):
        # a_rows, a_cols, b_rows, b_cols are lists of global indices
        # Local indices:
        # A: a_rows[0], a_rows[1] vs a_cols[0], a_cols[1]
        # B: b_rows[0], b_rows[1] vs b_cols[0], b_cols[1]
        
        # m1: (A00 + A11) * (B00 + B11)
        U[a_rows[0]*m + a_cols[0]][start_col] += 1
        U[a_rows[1]*m + a_cols[1]][start_col] += 1
        V[b_rows[0]*p + b_cols[0]][start_col] += 1
        V[b_rows[1]*p + b_cols[1]][start_col] += 1
        # W outputs:
        # C00 = m1 + ...
        # C00 is at W[b_cols[0]*n + a_rows[0]]
        # C01 is at W[b_cols[1]*n + a_rows[0]]
        # C10 is at W[b_cols[0]*n + a_rows[1]]
        # C11 is at W[b_cols[1]*n + a_rows[1]]
        
        # We will add W contributions in a separate step or directly?
        # Better to store multipliers and then combine.
        # But here we are adding to global U,V.
        # We need to add to W as well.
        
        # Let's just update U, V, W directly for each multiplier.
        
        # m1
        U[a_rows[0]*m + a_cols[0]][start_col] += 1
        U[a_rows[1]*m + a_cols[1]][start_col] += 1
        V[b_rows[0]*p + b_cols[0]][start_col] += 1
        V[b_rows[1]*p + b_cols[1]][start_col] += 1
        W[b_cols[0]*n + a_rows[0]][start_col] += 1 # C00
        W[b_cols[1]*n + a_rows[1]][start_col] += 1 # C11
        
        # m2: (A10 + A11) * B00
        U[a_rows[1]*m + a_cols[0]][start_col+1] += 1
        U[a_rows[1]*m + a_cols[1]][start_col+1] += 1
        V[b_rows[0]*p + b_cols[0]][start_col+1] += 1
        W[b_cols[0]*n + a_rows[1]][start_col+1] += 1 # C10
        
        # m3: A00 * (B01 - B11)
        U[a_rows[0]*m + a_cols[0]][start_col+2] += 1
        V[b_rows[0]*p + b_cols[1]][start_col+2] += 1
        V[b_rows[1]*p + b_cols[1]][start_col+2] -= 1
        W[b_cols[1]*n + a_rows[0]][start_col+2] += 1 # C01
        W[b_cols[1]*n + a_rows[1]][start_col+2] += 1 # C11
        
        # m4: A11 * (B10 - B00)
        U[a_rows[1]*m + a_cols[1]][start_col+3] += 1
        V[b_rows[1]*p + b_cols[0]][start_col+3] += 1
        V[b_rows[0]*p + b_cols[0]][start_col+3] -= 1
        W[b_cols[0]*n + a_rows[0]][start_col+3] += 1 # C00
        W[b_cols[0]*n + a_rows[1]][start_col+3] += 1 # C10
        
        # m5: (A00 + A01) * B11
        U[a_rows[0]*m + a_cols[0]][start_col+4] += 1
        U[a_rows[0]*m + a_cols[1]][start_col+4] += 1
        V[b_rows[1]*p + b_cols[1]][start_col+4] += 1
        W[b_cols[0]*n + a_rows[0]][start_col+4] -= 1 # C00
        W[b_cols[1]*n + a_rows[0]][start_col+4] += 1 # C01
        
        # m6: (A10 - A00) * (B00 + B10)
        U[a_rows[1]*m + a_cols[0]][start_col+5] += 1
        U[a_rows[0]*m + a_cols[0]][start_col+5] -= 1
        V[b_rows[0]*p + b_cols[0]][start_col+5] += 1
        V[b_rows[1]*p + b_cols[0]][start_col+5] += 1
        W[b_cols[1]*n + a_rows[1]][start_col+5] += 1 # C11
        
        # m7: (A01 - A11) * (B01 + B11)
        U[a_rows[0]*m + a_cols[1]][start_col+6] += 1
        U[a_rows[1]*m + a_cols[1]][start_col+6] -= 1
        V[b_rows[0]*p + b_cols[1]][start_col+6] += 1
        V[b_rows[1]*p + b_cols[1]][start_col+6] += 1
        W[b_cols[0]*n + a_rows[0]][start_col+6] += 1 # C00
        W[b_cols[1]*n + a_rows[1]][start_col+6] += 1 # C11
        
        return start_col + 7

    # Define Rank 4 block addition
    def add_rank4(a_rows, a_cols, b_rows, b_cols, start_col):
        # b_cols is single element
        k = b_cols[0]
        
        # m1: A00 * B00
        U[a_rows[0]*m + a_cols[0]][start_col] += 1
        V[b_rows[0]*p + k][start_col] += 1
        W[k*n + a_rows[0]][start_col] += 1 # C00
        
        # m2: A01 * B10
        U[a_rows[0]*m + a_cols[1]][start_col+1] += 1
        V[b_rows[1]*p + k][start_col+1] += 1
        W[k*n + a_rows[0]][start_col+1] += 1 # C00
        
        # m3: A10 * B00
        U[a_rows[1]*m + a_cols[0]][start_col+2] += 1
        V[b_rows[0]*p + k][start_col+2] += 1
        W[k*n + a_rows[1]][start_col+2] += 1 # C10
        
        # m4: A11 * B10
        U[a_rows[1]*m + a_cols[1]][start_col+3] += 1
        V[b_rows[1]*p + k][start_col+3] += 1
        W[k*n + a_rows[1]][start_col+3] += 1 # C10
        
        return start_col + 4

    # Block 1: A0 (rows 0,1; cols 0,1) x B0 (rows 0,1; cols 0..4)
    a_rows = [0, 1]
    a_cols_0 = [0, 1]
    b_rows_0 = [0, 1]
    
    # Sub-block 1.1: k=0,1
    col = add_strassen(a_rows, a_cols_0, b_rows_0, [0, 1], col)
    # Sub-block 1.2: k=2,3
    col = add_strassen(a_rows, a_cols_0, b_rows_0, [2, 3], col)
    # Sub-block 1.3: k=4
    col = add_rank4(a_rows, a_cols_0, b_rows_0, [4], col)
    
    # Block 2: A1 (rows 0,1; cols 2,3) x B1 (rows 2,3; cols 0..4)
    a_cols_1 = [2, 3]
    b_rows_1 = [2, 3]
    
    # Sub-block 2.1: k=0,1
    col = add_strassen(a_rows, a_cols_1, b_rows_1, [0, 1], col)
    # Sub-block 2.2: k=2,3
    col = add_strassen(a_rows, a_cols_1, b_rows_1, [2, 3], col)
    # Sub-block 2.3: k=4
    col = add_rank4(a_rows, a_cols_1, b_rows_1, [4], col)
    
    return {"U": U, "V": V, "W": W}
