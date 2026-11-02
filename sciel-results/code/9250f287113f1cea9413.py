def construct():
    """
    Rank-36 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split A into top 2 rows (A_top: 2x4) and bottom 1 row (A_bot: 1x4)
    - Split B into left 2 columns (B_left: 4x2) and right 2 columns (B_right: 4x2)
    - For top part (2x4 * 4x2):
      * C_top_left = A_TL * B_TL + A_TR * B_BL (two 2x2x2 Strassen, rank 7 each = 14)
      * C_top_right = A_TL * B_TR + A_TR * B_BR (two 2x2x2 Strassen, rank 7 each = 14)
    - For bottom part (1x4 * 4x2):
      * C_bot_left = schoolbook (rank 4)
      * C_bot_right = schoolbook (rank 4)
    - Total rank: 14 + 14 + 4 + 4 = 36
    """
    n, m, p = 3, 4, 4
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 12 x 36
    V = [[0]*rank for _ in range(m*p)]  # 16 x 36
    W = [[0]*rank for _ in range(p*n)]  # 12 x 36
    
    col = 0
    
    def strassen_2x2x2(a_row0, a_row1, a_col0, a_col1, b_row0, b_row1, b_col0, b_col1, c_row0, c_row1, c_col0, c_col1, c_idx):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A block: rows a_row0, a_row1; cols a_col0, a_col1
        B block: rows b_row0, b_row1; cols b_col0, b_col1  
        C block: rows c_row0, c_row1; cols c_col0, c_col1
        Uses 7 rank-1 terms starting at column c_idx.
        Returns c_idx + 7
        """
        # m1 = (a00 + a11)(b00 + b11)
        U[a_row0*m + a_col0][c_idx] = 1
        U[a_row1*m + a_col1][c_idx] = 1
        V[b_row0*p + b_col0][c_idx] = 1
        V[b_row1*p + b_col1][c_idx] = 1
        W[c_row0*n + c_col0][c_idx] = 1
        W[c_row1*n + c_col1][c_idx] = 1
        c_idx += 1
        
        # m2 = (a10 + a11) * b00
        U[a_row1*m + a_col0][c_idx] = 1
        U[a_row1*m + a_col1][c_idx] = 1
        V[b_row0*p + b_col0][c_idx] = 1
        W[c_row1*n + c_col0][c_idx] = 1
        W[c_row1*n + c_col1][c_idx] = -1
        c_idx += 1
        
        # m3 = a00 * (b01 - b11)
        U[a_row0*m + a_col0][c_idx] = 1
        V[b_row0*p + b_col1][c_idx] = 1
        V[b_row1*p + b_col1][c_idx] = -1
        W[c_row0*n + c_col1][c_idx] = 1
        W[c_row1*n + c_col1][c_idx] = 1
        c_idx += 1
        
        # m4 = a11 * (b10 - b00)
        U[a_row1*m + a_col1][c_idx] = 1
        V[b_row1*p + b_col0][c_idx] = 1
        V[b_row0*p + b_col0][c_idx] = -1
        W[c_row0*n + c_col0][c_idx] = 1
        W[c_row1*n + c_col0][c_idx] = 1
        c_idx += 1
        
        # m5 = (a00 + a01) * b11
        U[a_row0*m + a_col0][c_idx] = 1
        U[a_row0*m + a_col1][c_idx] = 1
        V[b_row1*p + b_col1][c_idx] = 1
        W[c_row0*n + c_col0][c_idx] = -1
        W[c_row0*n + c_col1][c_idx] = 1
        c_idx += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[a_row1*m + a_col0][c_idx] = 1
        U[a_row0*m + a_col0][c_idx] = -1
        V[b_row0*p + b_col0][c_idx] = 1
        V[b_row0*p + b_col1][c_idx] = 1
        W[c_row1*n + c_col1][c_idx] = 1
        c_idx += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[a_row0*m + a_col1][c_idx] = 1
        U[a_row1*m + a_col1][c_idx] = -1
        V[b_row1*p + b_col0][c_idx] = 1
        V[b_row1*p + b_col1][c_idx] = 1
        W[c_row0*n + c_col0][c_idx] = 1
        c_idx += 1
        
        return c_idx
    
    # Top part: A rows 0,1 (2x4) * B (4x2 for left, 4x2 for right)
    # C_top_left: rows 0,1 of C, cols 0,1
    # = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    
    # A_TL * B_TL: A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C rows 0,1 cols 0,1
    col = strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, col)
    
    # A_TR * B_BL: A rows 0,1 cols 2,3; B rows 2,3 cols 0,1; C rows 0,1 cols 0,1
    col = strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, col)
    
    # C_top_right: rows 0,1 of C, cols 2,3
    # = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # A_TL * B_TR: A rows 0,1 cols 0,1; B rows 0,1 cols 2,3; C rows 0,1 cols 2,3
    col = strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, col)
    
    # A_TR * B_BR: A rows 0,1 cols 2,3; B rows 2,3 cols 2,3; C rows 0,1 cols 2,3
    col = strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, col)
    
    # Bottom part: A row 2 (1x4) * B (4x2 for left, 4x2 for right)
    # C_bot_left: row 2 of C, cols 0,1
    for j in range(4):
        for k in range(2):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # C_bot_right: row 2 of C, cols 2,3
    for j in range(4):
        for k in range(2, 4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
