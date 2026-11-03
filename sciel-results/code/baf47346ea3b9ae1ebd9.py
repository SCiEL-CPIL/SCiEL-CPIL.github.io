import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        for (i, j), coeff in a_coeffs.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in b_coeffs.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in c_coeffs.items():
            W[k * n + i][col] += coeff
        col += 1

    def strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: indices for A block (2x2)
        # b_rows, b_cols: indices for B block (2x2)
        # c_rows, c_cols: indices for C block (2x2)
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_cols[0], c_rows[0]) # W index is k*n+i -> (col, row)
        c01 = (c_cols[1], c_rows[0])
        c10 = (c_cols[0], c_rows[1])
        c11 = (c_cols[1], c_rows[1])
        
        # M1 = (a00 + a10) (b00 + b10)
        add_term({a00: 1, a10: 1}, {b00: 1, b10: 1}, {c00: 1, c11: 1})
        
        # M2 = (a10 + a11) b00
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        
        # M3 = a00 (b01 - b11)
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        
        # M4 = a10 (b10 - b00)
        add_term({a10: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        
        # M5 = (a00 + a01) b11
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        
        # M6 = (a10 - a00) (b00 + b01)
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7 = (a01 - a10) (b10 + b11)
        add_term({a01: 1, a10: -1}, {b10: 1, b11: 1}, {c00: 1})

    def schoolbook_221_block(a_rows, a_cols, b_rows, b_col, c_row, c_col):
        # a_rows, a_cols: 2x2 block of A
        # b_rows, b_col: 2x1 block of B
        # c_row, c_col: 2x1 block of C
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b0 = (b_rows[0], b_col)
        b1 = (b_rows[1], b_col)
        
        c0 = (c_col, c_row[0])
        c1 = (c_col, c_row[1])
        
        # a00 * b0 -> c0
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        # a01 * b1 -> c0
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        # a10 * b0 -> c1
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        # a11 * b1 -> c1
        add_term({a11: 1}, {b1: 1}, {c1: 1})

    # A_L: rows 0,1; cols 0,1
    # A_R: rows 0,1; cols 2,3
    
    # B_T: rows 0,1; cols 0,1,2,3,4
    # B_B: rows 2,3; cols 0,1,2,3,4
    
    # C: rows 0,1; cols 0,1,2,3,4
    
    # Group 1: A_L * B_T,1 (cols 0,1)
    strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Group 2: A_L * B_T,2 (cols 2,3)
    strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # Group 3: A_L * B_T,3 (col 4)
    schoolbook_221_block([0,1], [0,1], [0,1], 4, [0,1], 4)
    
    # Group 4: A_R * B_B,1 (cols 0,1)
    strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Group 5: A_R * B_B,2 (cols 2,3)
    strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Group 6: A_R * B_B,3 (col 4)
    schoolbook_221_block([0,1], [2,3], [2,3], 4, [0,1], 4)
    
    return {"U": U, "V": V, "W": W}
