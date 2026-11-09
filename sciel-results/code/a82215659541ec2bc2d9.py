def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col, col_start):
        # A block: 2x2 at (a_row, a_col)
        # B block: 2x2 at (b_row, b_col)
        # C block: 2x2 at (c_row, c_col)
        
        # Strassen's 7 products
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10-b11)
        # M4 = a11*(b01-b10)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b10)
        # M7 = (a01-a11)*(b01+b11)
        
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        col = col_start
        U[a_row * m + a_col][col] = 1        # a00
        U[(a_row+1) * m + (a_col+1)][col] = 1  # a11
        V[b_row * p + b_col][col] = 1        # b00
        V[(b_row+1) * p + (b_col+1)][col] = 1  # b11
        W[c_col * n + c_row][col] = 1        # c00
        W[(c_col+1) * n + (c_row+1)][col] = 1  # c11
        
        # M2
        col = col_start + 1
        U[(a_row+1) * m + a_col][col] = 1    # a10
        U[(a_row+1) * m + (a_col+1)][col] = 1  # a11
        V[b_row * p + b_col][col] = 1        # b00
        W[c_col * n + (c_row+1)][col] = 1    # c10
        W[(c_col+1) * n + (c_row+1)][col] = -1  # c11
        
        # M3
        col = col_start + 2
        U[a_row * m + a_col][col] = 1        # a00
        V[(b_row+1) * p + b_col][col] = 1    # b10
        V[(b_row+1) * p + (b_col+1)][col] = -1  # b11
        W[(c_col+1) * n + c_row][col] = 1    # c01
        W[(c_col+1) * n + (c_row+1)][col] = 1  # c11
        
        # M4
        col = col_start + 3
        U[(a_row+1) * m + (a_col+1)][col] = 1  # a11
        V[b_row * p + (b_col+1)][col] = 1    # b01
        V[(b_row+1) * p + b_col][col] = -1    # b10
        W[c_col * n + c_row][col] = 1        # c00
        W[c_col * n + (c_row+1)][col] = 1    # c10
        
        # M5
        col = col_start + 4
        U[a_row * m + a_col][col] = 1        # a00
        U[a_row * m + (a_col+1)][col] = 1    # a01
        V[(b_row+1) * p + (b_col+1)][col] = 1  # b11
        W[(c_col+1) * n + c_row][col] = 1    # c01
        W[c_col * n + c_row][col] = -1        # c00
        
        # M6
        col = col_start + 5
        U[(a_row+1) * m + a_col][col] = 1    # a10
        U[a_row * m + a_col][col] = -1        # a00
        V[b_row * p + b_col][col] = 1        # b00
        V[(b_row+1) * p + b_col][col] = 1    # b10
        W[(c_col+1) * n + (c_row+1)][col] = 1  # c11
        
        # M7
        col = col_start + 6
        U[a_row * m + (a_col+1)][col] = 1    # a01
        U[(a_row+1) * m + (a_col+1)][col] = -1  # a11
        V[b_row * p + (b_col+1)][col] = 1    # b01
        V[(b_row+1) * p + (b_col+1)][col] = 1  # b11
        W[c_col * n + c_row][col] = 1        # c00
    
    # Block decomposition:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-5)
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-5)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C02 (rows 0-1, cols 4-5)
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3), C12 (rows 2-3, cols 4-5)
    
    # C00 = A00*B00 + A01*B10
    col = 0
    add_strassen(0, 0, 0, 0, 0, 0, col)      # A00*B00
    col += 7
    add_strassen(0, 2, 2, 0, 0, 0, col)      # A01*B10
    col += 7
    
    # C01 = A00*B01 + A01*B11
    add_strassen(0, 0, 0, 2, 0, 2, col)      # A00*B01
    col += 7
    add_strassen(0, 2, 2, 2, 0, 2, col)      # A01*B11
    col += 7
    
    # C02 = A00*B02 + A01*B12
    add_strassen(0, 0, 0, 4, 0, 4, col)      # A00*B02
    col += 7
    add_strassen(0, 2, 2, 4, 0, 4, col)      # A01*B12
    col += 7
    
    # C10 = A10*B00 + A11*B10
    add_strassen(2, 0, 0, 0, 2, 0, col)      # A10*B00
    col += 7
    add_strassen(2, 2, 2, 0, 2, 0, col)      # A11*B10
    col += 7
    
    # C11 = A10*B01 + A11*B11
    add_strassen(2, 0, 0, 2, 2, 2, col)      # A10*B01
    col += 7
    add_strassen(2, 2, 2, 2, 2, 2, col)      # A11*B11
    col += 7
    
    # C12 = A10*B02 + A11*B12
    add_strassen(2, 0, 0, 4, 2, 4, col)      # A10*B02
    col += 7
    add_strassen(2, 2, 2, 4, 2, 4, col)      # A11*B12
    col += 7
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
