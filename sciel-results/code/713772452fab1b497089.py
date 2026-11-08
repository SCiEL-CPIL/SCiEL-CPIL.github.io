"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using Strassen on 2x2 blocks: 6 block products, each rank 7 => total rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2 multiplication coefficients
    # For X*Y where X, Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # C00 = M1 - M5 + M3 + M4
    # C01 = M5 + M7
    # C10 = M6 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Each product (X, Y) contributes 7 terms
    # X is a 2x2 block of A, Y is a 2x2 block of B
    
    # Define the 6 block products:
    # (A_L, B_11), (A_L, B_12), (A_L, B_13), (A_R, B_21), (A_R, B_22), (A_R, B_23)
    
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_11 = B[0:2, 0:2], B_12 = B[0:2, 2:4], B_13 = B[0:2, 4:6]
    # B_21 = B[2:4, 0:2], B_22 = B[2:4, 2:4], B_23 = B[2:4, 4:6]
    
    # For a block product X*Y:
    # X indices in A: row offset x_r, col offset x_c (each 2x2)
    # Y indices in B: row offset y_r, col offset y_k (each 2x2)
    # Result goes to C at row offset x_r, col offset y_k
    
    # Strassen term definitions: (coeff for X part, coeff for Y part)
    # Each term is sum of (a*x_idx, b*y_idx) pairs for X and Y
    
    # For X (2x2), elements at positions (0,0), (0,1), (1,0), (1,1)
    # For Y (2x2), elements at positions (0,0), (0,1), (1,0), (1,1)
    
    # M1: X part: (0,0)+(1,1), Y part: (0,0)+(1,1)
    # M2: X part: (1,0)+(1,1), Y part: (0,0)
    # M3: X part: (0,0), Y part: (0,1)-(1,1)
    # M4: X part: (1,1), Y part: (1,0)-(0,0)
    # M5: X part: (0,0)+(0,1), Y part: (1,1)
    # M6: X part: (1,0)-(0,0), Y part: (0,0)+(0,1)
    # M7: X part: (0,1)-(1,1), Y part: (1,0)+(1,1)
    
    # Output coefficients:
    # C00: M1: +1, M5: -1, M3: +1, M4: +1
    # C01: M5: +1, M7: +1
    # C10: M6: +1, M4: +1
    # C11: M1: +1, M2: -1, M6: +1, M7: +1
    
    # Let's define the Strassen terms as lists of (x_pos, coeff) and (y_pos, coeff)
    # x_pos, y_pos in {(0,0), (0,1), (1,0), (1,1)}
    
    strassen_terms = [
        # (x_factors, y_factors) where each is list of (pos, coeff)
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),  # M1
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),                # M2
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),               # M3
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),               # M4
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),                # M5
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),   # M6
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),   # M7
    ]
    
    # Output mapping: for each output position (c_r, c_c) in 2x2, which terms with what coeff
    # C00: M1:+1, M5:-1, M3:+1, M4:+1
    # C01: M5:+1, M7:+1
    # C10: M6:+1, M4:+1
    # C11: M1:+1, M2:-1, M6:+1, M7:+1
    output_map = {
        (0,0): [(0, 1), (4, -1), (2, 1), (3, 1)],
        (0,1): [(4, 1), (6, 1)],
        (1,0): [(5, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (5, 1), (6, 1)],
    }
    
    # The 6 block products: (x_r, x_c, y_r, y_k, c_r, c_k)
    # x_r, x_c: row/col offset of X in A
    # y_r, y_k: row/col offset of Y in B
    # c_r, c_k: row/col offset of result in C
    block_products = [
        (0, 0, 0, 0, 0, 0),  # A_L * B_11 -> C[:,0:2]
        (0, 0, 0, 2, 0, 2),  # A_L * B_12 -> C[:,2:4]
        (0, 0, 0, 4, 0, 4),  # A_L * B_13 -> C[:,4:6]
        (0, 2, 2, 0, 0, 0),  # A_R * B_21 -> C[:,0:2]
        (0, 2, 2, 2, 0, 2),  # A_R * B_22 -> C[:,2:4]
        (0, 2, 2, 4, 0, 4),  # A_R * B_23 -> C[:,4:6]
    ]
    
    col = 0
    for (x_r, x_c, y_r, y_k, c_r, c_k) in block_products:
        for t in range(7):
            x_factors, y_factors = strassen_terms[t]
            
            # Fill U: for each (x_pos, coeff) in x_factors
            for (xr, xc), coeff in x_factors:
                # Global index in A: row = x_r + xr, col = x_c + xc
                a_row = x_r + xr
                a_col = x_c + xc
                u_idx = a_row * m + a_col
                U[u_idx][col] += coeff
            
            # Fill V: for each (y_pos, coeff) in y_factors
            for (yr, yk), coeff in y_factors:
                # Global index in B: row = y_r + yr, col = y_k + yk
                b_row = y_r + yr
                b_col = y_k + yk
                v_idx = b_row * p + b_col
                V[v_idx][col] += coeff
            
            # Fill W: for each output position, add contribution
            for (cr, cc), terms in output_map.items():
                for (term_idx, term_coeff) in terms:
                    if term_idx == t:
                        # Global index in C: row = c_r + cr, col = c_k + cc
                        c_row = c_r + cr
                        c_col = c_k + cc
                        # W index: k*n + i where k=c_col, i=c_row
                        w_idx = c_col * n + c_row
                        W[w_idx][col] += term_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
