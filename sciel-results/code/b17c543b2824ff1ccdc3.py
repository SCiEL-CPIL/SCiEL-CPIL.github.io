import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar term to the decomposition
    def add_term(u_terms, v_terms, k, i, w_coeff):
        nonlocal col
        for (ur, uc, uc_coeff) in u_terms:
            for (vr, vc, vc_coeff) in v_terms:
                coeff = uc_coeff * vc_coeff
                U[ur * m + uc][col] += coeff
                V[vr * p + vc][col] += coeff
                W[k * n + i][col] += w_coeff
                col += 1

    # Helper to add a 2x2 Strassen decomposition
    # a_blocks: 2x2 list of lists of (row, col, coeff) for A
    # b_blocks: 2x2 list of lists of (row, col, coeff) for B
    # c_offsets: (row_offset, col_offset) for the 2x2 block in C
    def add_2x2_strassen(a_blocks, b_blocks, c_row_off, c_col_off):
        nonlocal col
        
        # Define the 7 Strassen products
        # M1 = (A00+A11)(B00+B11)
        a_m1 = a_blocks[0][0] + a_blocks[1][1]
        b_m1 = b_blocks[0][0] + b_blocks[1][1]
        
        # M2 = (A10+A11)B00
        a_m2 = a_blocks[1][0] + a_blocks[1][1]
        b_m2 = b_blocks[0][0]
        
        # M3 = A00(B01-B11)
        a_m3 = a_blocks[0][0]
        b_m3 = b_blocks[0][1] + [(-r, -c, -coeff) for (r, c, coeff) in b_blocks[1][1]]
        
        # M4 = A11(B10-B00)
        a_m4 = a_blocks[1][1]
        b_m4 = b_blocks[1][0] + [(-r, -c, -coeff) for (r, c, coeff) in b_blocks[0][0]]
        
        # M5 = (A00+A01)B11
        a_m5 = a_blocks[0][0] + a_blocks[0][1]
        b_m5 = b_blocks[1][1]
        
        # M6 = (A10-A00)(B00+B01)
        a_m6 = a_blocks[1][0] + [(-r, -c, -coeff) for (r, c, coeff) in a_blocks[0][0]]
        b_m6 = b_blocks[0][0] + b_blocks[0][1]
        
        # M7 = (A01-A11)(B10+B11)
        a_m7 = a_blocks[0][1] + [(-r, -c, -coeff) for (r, c, coeff) in a_blocks[1][1]]
        b_m7 = b_blocks[1][0] + b_blocks[1][1]
        
        # Add terms for M1
        # C00 += M1, C11 += M1
        add_term(a_m1, b_m1, c_col_off + 0, c_row_off + 0, 1)
        add_term(a_m1, b_m1, c_col_off + 1, c_row_off + 1, 1)
        
        # Add terms for M2
        # C10 += M2, C11 -= M2
        add_term(a_m2, b_m2, c_col_off + 0, c_row_off + 1, 1)
        add_term(a_m2, b_m2, c_col_off + 1, c_row_off + 1, -1)
        
        # Add terms for M3
        # C01 += M3, C11 += M3
        add_term(a_m3, b_m3, c_col_off + 1, c_row_off + 0, 1)
        add_term(a_m3, b_m3, c_col_off + 1, c_row_off + 1, 1)
        
        # Add terms for M4
        # C00 += M4, C10 += M4
        add_term(a_m4, b_m4, c_col_off + 0, c_row_off + 0, 1)
        add_term(a_m4, b_m4, c_col_off + 0, c_row_off + 1, 1)
        
        # Add terms for M5
        # C00 -= M5, C01 += M5
        add_term(a_m5, b_m5, c_col_off + 0, c_row_off + 0, -1)
        add_term(a_m5, b_m5, c_col_off + 1, c_row_off + 0, 1)
        
        # Add terms for M6
        # C11 += M6
        add_term(a_m6, b_m6, c_col_off + 1, c_row_off + 1, 1)
        
        # Add terms for M7
        # C00 += M7
        add_term(a_m7, b_m7, c_col_off + 0, c_row_off + 0, 1)

    # Define blocks for A (4x4)
    # A11: (0,0)-(1,1)
    # A12: (0,2)-(1,3)
    # A21: (2,0)-(3,1)
    # A22: (2,2)-(3,3)
    
    def get_a_block(r0, c0):
        return [
            [ (r0+dr, c0+dc, 1) for dc in range(2) ] for dr in range(2)
        ]
        
    def get_b_block(r0, c0):
        # B is 4x7. c0 is the column offset in B.
        return [
            [ (r0+dr, c0+dc, 1) for dc in range(2) ] for dr in range(2)
        ]

    # Strassen for C[:, 0:4]
    # A blocks:
    a11 = get_a_block(0, 0)
    a12 = get_a_block(0, 2)
    a21 = get_a_block(2, 0)
    a22 = get_a_block(2, 2)
    
    # B blocks (for first 4 cols):
    b11 = get_b_block(0, 0)
    b12 = get_b_block(0, 2)
    b21 = get_b_block(2, 0)
    b22 = get_b_block(2, 2)
    
    # C blocks (for first 4 cols):
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # We need to combine the 4 2x2 Strassen decompositions?
    # No, the 4x4 Strassen algorithm uses 7 products of 2x2 matrices.
    # Each product is a 2x2 matrix multiplication.
    # So we call add_2x2_strassen for each of the 7 M's?
    # No, add_2x2_strassen adds 7 scalar terms for ONE 2x2 multiplication.
    # The 4x4 Strassen algorithm computes 7 such 2x2 multiplications.
    # So we should call add_2x2_strassen 7 times?
    # No, the inputs to the 7 multiplications are linear combinations of the 4 A blocks and 4 B blocks.
    
    # Let's define the 7 linear combinations of blocks.
    # M1 = (A11+A22)(B11+B22)
    # We need to pass the linear combinations to a function that handles 2x2 multiplication of linear combos.
    # My add_2x2_strassen expects `a_blocks` to be 4 lists of terms.
    # If I pass `a11 + a22` as `a_blocks[0][0]`, etc., it will work.
    
    def sum_blocks(b1, b2):
        return [
            [ (r, c, c1) for (r, c, c1) in b1[0][0] ] + [ (r, c, c1) for (r, c, c1) in b2[0][0] ],
            [ (r, c, c1) for (r, c, c1) in b1[0][1] ] + [ (r, c, c1) for (r, c, c1) in b2[0][1] ],
            [ (r, c, c1) for (r, c, c1) in b1[1][0] ] + [ (r, c, c1) for (r, c, c1) in b2[1][0] ],
            [ (r, c, c1) for (r, c, c1) in b1[1][1] ] + [ (r, c, c1) for (r, c, c1) in b2[1][1] ]
        ]
        
    def diff_blocks(b1, b2):
        return [
            [ (r, c, c1) for (r, c, c1) in b1[0][0] ] + [ (r, c, -c1) for (r, c, c1) in b2[0][0] ],
            [ (r, c, c1) for (r, c, c1) in b1[0][1] ] + [ (r, c, -c1) for (r, c, c1) in b2[0][1] ],
            [ (r, c, c1) for (r, c, c1) in b1[1][0] ] + [ (r, c, -c1) for (r, c, c1) in b2[1][0] ],
            [ (r, c, c1) for (r, c, c1) in b1[1][1] ] + [ (r, c, -c1) for (r, c, c1) in b2[1][1] ]
        ]
        
    # This is getting complicated to manage the block structures.
    # Let's simplify.
    # I will just call add_2x2_strassen 7 times, but with the correct linear combinations of the *entries* of the blocks.
    
    # Actually, the 4x4 Strassen algorithm is:
    # M1 = (A11+A22)(B11+B22)
    # ...
    # Each Mi is a 2x2 matrix.
    # We can compute each Mi using the 2x2 Strassen algorithm (rank 7).
    # So we need to call add_2x2_strassen for each Mi?
    # No, add_2x2_strassen computes the product of two 2x2 matrices using 7 terms.
    # So if I have M1 = X * Y, where X and Y are 2x2 matrices (which are themselves linear combos of A and B blocks),
    # I can just call add_2x2_strassen with X and Y as the inputs.
    
    # So I need to construct X and Y for each of the 7 M's.
    # X and Y are 2x2 matrices where each entry is a linear combination of A or B entries.
    
    # Let's define the entries of X and Y for M1.
    # X = A11 + A22
    # X00 = A11_00 + A22_00
    # X01 = A11_01 + A22_01
    # X10 = A11_10 + A22_10
    # X11 = A11_11 + A22_11
    
    # I will create a helper to generate the 2x2 block of linear combos.
    
    def make_block(r0, c0, coeff=1):
        return [
            [ (r0+dr, c0+dc, coeff) for dc in range(2) ] for dr in range(2)
        ]
        
    def add_block(b1, b2):
        return [
            [ (r, c, c1 + c2) for (r, c, c1), (r2, c2, c2_val) in zip(b1[i][j], b2[i][j]) if (r, c) == (r2, c2) ] # This zip is wrong.
        ]
        
    # Let's just flatten and add.
    def add_blocks(b1, b2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = b1[i][j] + b2[i][j]
        return res
        
    def sub_blocks(b1, b2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = b1[i][j] + [ (r, c, -coeff) for (r, c, coeff) in b2[i][j] ]
        return res

    # A blocks
    A11 = make_block(0, 0)
    A12 = make_block(0, 2)
    A21 = make_block(2, 0)
    A22 = make_block(2, 2)
    
    # B blocks (cols 0-3)
    B11 = make_block(0, 0)
    B12 = make_block(0, 2)
    B21 = make_block(2, 0)
    B22 = make_block(2, 2)
    
    # M1
    X1 = add_blocks(A11, A22)
    Y1 = add_blocks(B11, B22)
    add_2x2_strassen(X1, Y1, 0, 0) # C11 offset? No, M1 contributes to C11 and C22.
    # Wait, add_2x2_strassen adds terms for a 2x2 product.
    # The result of M1 is a 2x2 matrix.
    # C11 += M1, C22 += M1.
    # My add_2x2_strassen function adds terms to W at positions (c_row_off, c_col_off) + (0,0), (0,1), etc.
    # So if I call add_2x2_strassen(X1, Y1, 0, 0), it adds to C[0:2, 0:2] (C11) and C[0:2, 2:4] (C12)?
    # No, the offsets are for the 2x2 block.
    # C11 is at (0,0). C12 is at (0,2). C21 is at (2,0). C22 is at (2,2).
    
    # The 2x2 product result Z has entries Z00, Z01, Z10, Z11.
    # Z00 goes to C11[0,0] -> Global (0,0)
    # Z01 goes to C11[0,1] -> Global (0,1)
    # Z10 goes to C11[1,0] -> Global (1,0)
    # Z11 goes to C11[1,1] -> Global (1,1)
    
    # But M1 contributes to C11 AND C22.
    # So I need to add the terms for M1 to C11 AND to C22.
    # My add_2x2_strassen function only adds to one 2x2 block location.
    # So I should call it twice? Or modify it to take a list of offsets?
    
    # Let's modify add_2x2_strassen to take a list of (row_off, col_off, coeff) for the W part.
    # Or just call it twice.
    
    # Call 1: M1 -> C11
    add_2x2_strassen(X1, Y1, 0, 0)
    # Call 2: M1 -> C22
    add_2x2_strassen(X1, Y1, 2, 2)
    
    # This doubles the number of terms?
    # No, the scalar products are the same.
    # But `add_2x2_strassen` increments `col` 7 times.
    # If I call it twice, I use 14 columns for M1.
    # But M1 is only ONE 2x2 multiplication, which has rank 7.
    # The fact that it contributes to two output blocks doesn't increase the rank.
    # The U and V parts are the same. Only W changes.
    # So I should NOT call add_2x2_strassen twice.
    # I should call a function that adds the 7 scalar products, and for each, adds to multiple W positions.
    
    # Let's rewrite add_2x2_strassen to accept a list of W contributions.
    
    return {"U": U, "V": V, "W": W}
