"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Recursive Strassen: 6x6 -> 2x2 blocks of 3x3 (7 terms)
    # Each 3x3 -> 2x2 blocks of sizes (2,1) (7 terms)
    # Each sub-product decomposed via schoolbook
    # Total rank = 7 * 7 * 4 = 196
    
    n = m = p = 6
    rank = 196
    U = [[0] * rank for _ in range(n * m)]  # 36 x 196
    V = [[0] * rank for _ in range(m * p)]  # 36 x 196
    W = [[0] * rank for _ in range(p * n)]  # 36 x 196
    
    # Level 1 Strassen for 6x6 = 2x2 blocks of 3x3
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 3x3
    #
    # T1 = (A00+A11)*(B00+B11)
    # T2 = (A10+A11)*B00
    # T3 = A00*(B01-B11)
    # T4 = A11*(B10-B00)
    # T5 = (A00+A01)*B11
    # T6 = (A10-A00)*(B00+B01)
    # T7 = (A01-A11)*(B10+B11)
    #
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # Level 2 Strassen for 3x3 = 2x2 blocks of sizes (2,1)
    # X = [[X00, X01], [X10, X11]], Y = [[Y00, Y01], [Y10, Y11]]
    # X00,Y00: 2x2; X01,Y01: 2x1; X10,Y10: 1x2; X11,Y11: 1x1
    #
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Level 1 Strassen terms: (left_blocks, right_blocks)
    # Each block is identified by (br, bc) where br in {0,1}, bc in {0,1}
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    L1_terms = [
        # T1: (A00+A11)*(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T2: (A10+A11)*B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # T3: A00*(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # T4: A11*(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # T5: (A00+A01)*B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # T6: (A10-A00)*(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # T7: (A01-A11)*(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Result block contributions: C_block = sum of signed T terms
    L1_result = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # Level 2 Strassen terms for 3x3 with blocks of sizes (2,1)
    # Within a 3x3 block, sub-blocks are:
    #   (0,0): rows 0-1, cols 0-1 (2x2)
    #   (0,1): rows 0-1, col 2 (2x1)
    #   (1,0): row 2, cols 0-1 (1x2)
    #   (1,1): row 2, col 2 (1x1)
    #
    # L2_terms: (left_subblocks, right_subblocks)
    L2_terms = [
        # M1: (X00+X11)*(Y00+Y11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (X10+X11)*Y00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # M3: X00*(Y01-Y11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # M4: X11*(Y10-Y00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # M5: (X00+X01)*Y11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # M6: (X10-X00)*(Y00+Y01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M7: (X01-X11)*(Y10+Y11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # L2 result: Z_block = sum of signed M terms
    L2_result = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # Sub-block dimensions within a 3x3:
    # (0,0): 2x2, (0,1): 2x1, (1,0): 1x2, (1,1): 1x1
    sub_dims = {
        (0,0): (2, 2),
        (0,1): (2, 1),
        (1,0): (1, 2),
        (1,1): (1, 1),
    }
    
    col = 0
    for l1_idx, (l1_left, l1_right) in enumerate(L1_terms):
        for l2_idx, (l2_left, l2_right) in enumerate(L2_terms):
            # For each level-2 term, we compute a product of two linear combinations
            # The left factor is a 3x3 matrix X, right factor is 3x3 matrix Y
            # X is formed from left sub-blocks of the level-1 left blocks
            # Y is formed from right sub-blocks of the level-1 right blocks
            #
            # We decompose X*Y into schoolbook scalar multiplications
            # For each (i, a, k) in the 3x3 product:
            #   x_{i,a} * y_{a,k}
            # where x_{i,a} is a linear combo of A entries, y_{a,k} is a linear combo of B entries
            
            for i in range(3):
                for a in range(3):
                    for k in range(3):
                        # Determine which sub-blocks contain position (i, a) in the 3x3
                        # and (a, k) in the 3x3
                        # For left factor X at position (i, a):
                        #   Find which sub-block (sr, sc) contains row i, col a
                        #   i in 0-1 -> sr=0, i=2 -> sr=1
                        #   a in 0-1 -> sc=0, a=2 -> sc=1
                        # But we need to check against l2_left to see which sub-blocks are included
                        
                        # For U: for each sub-block in l2_left that contains (i,a)
                        for (sr, sc), coeff_l2 in l2_left.items():
                            # Check if (i, a) is within sub-block (sr, sc)
                            sr_rows, sr_cols = sub_dims[(sr, sc)]
                            i_start = 0 if sr == 0 else 2
                            a_start = 0 if sc == 0 else 2
                            if i_start <= i < i_start + sr_rows and a_start <= a < a_start + sr_cols:
                                # This sub-block contributes to x_{i,a}
                                # Now for each level-1 block in l1_left that maps to this sub-block
                                for (l1_br, l1_bc), coeff_l1 in l1_left.items():
                                    # The actual row/col in the 6x6 A matrix
                                    row_A = 3 * l1_br + i
                                    col_A = 3 * l1_bc + a
                                    U[row_A * 6 + col_A][col] += coeff_l1 * coeff_l2
                        
                        # For V: for each sub-block in l2_right that contains (a, k)
                        for (sr, sc), coeff_l2 in l2_right.items():
                            sr_rows, sr_cols = sub_dims[(sr, sc)]
                            a_start = 0 if sr == 0 else 2
                            k_start = 0 if sc == 0 else 2
                            if a_start <= a < a_start + sr_rows and k_start <= k < k_start + sr_cols:
                                for (l1_br, l1_bc), coeff_l1 in l1_right.items():
                                    row_B = 3 * l1_br + a
                                    col_B = 3 * l1_bc + k
                                    V[row_B * 6 + col_B][col] += coeff_l1 * coeff_l2
                        
                        # For W: determine which output positions this term contributes to
                        # The level-2 term l2_idx contributes to output sub-blocks with signs from L2_result
                        # The level-1 term l1_idx contributes to output blocks with signs from L1_result
                        for (l2_rbr, l2_rbc), l2_signs in L2_result.items():
                            for (si2, sign2) in l2_signs:
                                if si2 == l2_idx:
                                    for (l1_rbr, l1_rbc), l1_signs in L1_result.items():
                                        for (si1, sign1) in l1_signs:
                                            if si1 == l1_idx:
                                                i_total = 3 * l1_rbr + i
                                                k_total = 3 * l1_rbc + k
                                                W_row = k_total * 6 + i_total
                                                W[W_row][col] += sign1 * sign2
                        
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
