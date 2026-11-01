import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # We partition the 11 columns into 5 blocks of 2 and 1 block of 1.
    # Blocks: [0,1], [2,3], [4,5], [6,7], [8,9], [10]
    
    # Total rank will be 5 * 7 + 4 = 39.
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Define Strassen terms for a 2x2 block
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices for block k: 
    #   col 0: 0: b10, 1: b20
    #   col 1: 0: b11, 1: b21
    # C indices for block k:
    #   col 0: 0: c10, 1: c20
    #   col 1: 0: c11, 1: c21
    
    def add_strassen_block(start_col):
        nonlocal col_offset
        # B indices
        b10_idx = 0 * p + start_col
        b20_idx = 1 * p + start_col
        b11_idx = 0 * p + (start_col + 1)
        b21_idx = 1 * p + (start_col + 1)
        
        # C indices
        c10_idx = start_col * n + 0
        c20_idx = start_col * n + 1
        c11_idx = (start_col + 1) * n + 0
        c21_idx = (start_col + 1) * n + 1
        
        # A indices
        a_idx = 0
        b_idx = 1
        c_idx = 2
        d_idx = 3
        
        # M1 = (a+d)(b10+b20+b11+b21) ? No.
        # M1 = (a+d)(e+h) where e=b10, h=b21?
        # Standard Strassen:
        # e=f? No.
        # B = [[e, f], [g, h]]
        # Here B_block = [[b10, b11], [b20, b21]]
        # So e=b10, f=b11, g=b20, h=b21.
        
        # M1 = (a+d)(e+h) = (a+d)(b10+b21)
        U[a_idx][col_offset] = 1
        U[d_idx][col_offset] = 1
        V[b10_idx][col_offset] = 1
        V[b21_idx][col_offset] = 1
        # C11 += M1, C22 += M1
        W[c10_idx][col_offset] = 1
        W[c21_idx][col_offset] = 1
        col_offset += 1
        
        # M2 = (c+d)g = (c+d)b20
        U[c_idx][col_offset] = 1
        U[d_idx][col_offset] = 1
        V[b20_idx][col_offset] = 1
        # C21 += M2, C22 -= M2
        W[c20_idx][col_offset] = 1
        W[c21_idx][col_offset] = -1
        col_offset += 1
        
        # M3 = a(f-h) = a(b11-b21)
        U[a_idx][col_offset] = 1
        V[b11_idx][col_offset] = 1
        V[b21_idx][col_offset] = -1
        # C12 += M3, C22 += M3
        W[c11_idx][col_offset] = 1
        W[c21_idx][col_offset] = 1
        col_offset += 1
        
        # M4 = d(e-g) = d(b10-b20)
        U[d_idx][col_offset] = 1
        V[b10_idx][col_offset] = 1
        V[b20_idx][col_offset] = -1
        # C11 += M4, C21 += M4
        W[c10_idx][col_offset] = 1
        W[c20_idx][col_offset] = 1
        col_offset += 1
        
        # M5 = (a+b)h = (a+b)b21
        U[a_idx][col_offset] = 1
        U[b_idx][col_offset] = 1
        V[b21_idx][col_offset] = 1
        # C11 -= M5, C12 += M5
        W[c10_idx][col_offset] = -1
        W[c11_idx][col_offset] = 1
        col_offset += 1
        
        # M6 = (c-a)(e+g) = (c-a)(b10+b20)
        U[c_idx][col_offset] = 1
        U[a_idx][col_offset] = -1
        V[b10_idx][col_offset] = 1
        V[b20_idx][col_offset] = 1
        # C22 += M6
        W[c21_idx][col_offset] = 1
        col_offset += 1
        
        # M7 = (b-d)(f-h) = (b-d)(b11-b21)
        U[b_idx][col_offset] = 1
        U[d_idx][col_offset] = -1
        V[b11_idx][col_offset] = 1
        V[b21_idx][col_offset] = -1
        # C11 += M7
        W[c10_idx][col_offset] = 1
        col_offset += 1

    def add_schoolbook_block(start_col):
        nonlocal col_offset
        b1_idx = 0 * p + start_col
        b2_idx = 1 * p + start_col
        c1_idx = start_col * n + 0
        c2_idx = start_col * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 2
        d_idx = 3
        
        # a * b1
        U[a_idx][col_offset] = 1
        V[b1_idx][col_offset] = 1
        W[c1_idx][col_offset] = 1
        col_offset += 1
        
        # b * b2
        U[b_idx][col_offset] = 1
        V[b2_idx][col_offset] = 1
        W[c1_idx][col_offset] = 1
        col_offset += 1
        
        # c * b1
        U[c_idx][col_offset] = 1
        V[b1_idx][col_offset] = 1
        W[c2_idx][col_offset] = 1
        col_offset += 1
        
        # d * b2
        U[d_idx][col_offset] = 1
        V[b2_idx][col_offset] = 1
        W[c2_idx][col_offset] = 1
        col_offset += 1

    # Apply blocks
    for i in range(5):
        start_col = i * 2
        add_strassen_block(start_col)
    
    add_schoolbook_block(10)
    
    return {"U": U, "V": V, "W": W}
