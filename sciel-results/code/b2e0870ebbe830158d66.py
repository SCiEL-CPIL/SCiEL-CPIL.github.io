"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    # 4x4 = 2x2 blocks of 2x2, each 2x2 block product done via Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 decomposition formulas:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Block-level Strassen for 4x4 (treating as 2x2 blocks):
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level term Mk, we compute a 2x2 matrix product using Strassen
    # The "X" in inner Strassen is a linear combination of A blocks
    # The "Y" in inner Strassen is a linear combination of B blocks
    
    # Define block-level Strassen terms: (list of (block_row, block_col, coeff) for A, same for B)
    block_terms = [
        # M1: (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3: A00(B10-B11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M4: A11(B01-B00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # M5: (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6: (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7: (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Inner Strassen terms for a 2x2 matrix product X*Y:
    # S1: (X00+X11)(Y00+Y11)
    # S2: (X10+X11)Y00
    # S3: X00(Y10-Y11)
    # S4: X11(Y01-Y00)
    # S5: (X00+X01)Y11
    # S6: (X10-X00)(Y00+Y01)
    # S7: (X01-X11)(Y10+Y11)
    
    # For each inner term, define the 2x2 indices (row, col) of X and Y that are used
    # and their coefficients
    inner_terms = [
        # S1: (X00+X11)(Y00+Y11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # S2: (X10+X11)Y00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # S3: X00(Y10-Y11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # S4: X11(Y01-Y00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # S5: (X00+X01)Y11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # S6: (X10-X00)(Y00+Y01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # S7: (X01-X11)(Y10+Y11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Output combination: how each inner Strassen result contributes to the 2x2 output
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    output_combos = [
        # For output position (0,0): coefficients for S1..S7
        [(0,0,1), (0,3,1), (0,4,-1), (0,6,1)],
        # For output position (0,1): coefficients for S1..S7
        [(0,2,1), (0,4,1)],
        # For output position (1,0): coefficients for S1..S7
        [(0,1,1), (0,3,1)],
        # For output position (1,1): coefficients for S1..S7
        [(0,0,1), (0,1,-1), (0,2,1), (0,5,1)],
    ]
    
    # Now build the 49 rank-1 terms
    col = 0
    for b_idx in range(7):
        A_block_comb, B_block_comb = block_terms[b_idx]
        
        for i_idx in range(7):
            X_comb, Y_comb = inner_terms[i_idx]
            
            # Build U vector: for each element in the combined A matrix
            # A_block_comb gives which blocks of A to use
            # X_comb gives which 2x2 positions within those blocks to use
            # Global index in A: i = 2*block_row + x_row, j = 2*block_col + x_col
            
            U_vec = [0] * (n * m)
            for (br, bc, bc_coeff) in A_block_comb:
                for (xr, xc, xc_coeff) in X_comb:
                    i = 2 * br + xr
                    j = 2 * bc + xc
                    idx = i * m + j
                    U_vec[idx] += bc_coeff * xc_coeff
            
            # Build V vector: for each element in the combined B matrix
            # B_block_comb gives which blocks of B to use
            # Y_comb gives which 2x2 positions within those blocks to use
            # Global index in B: i = 2*block_row + y_row, j = 2*block_col + y_col
            
            V_vec = [0] * (m * p)
            for (br, bc, bc_coeff) in B_block_comb:
                for (yr, yc, yc_coeff) in Y_comb:
                    i = 2 * br + yr
                    j = 2 * bc + yc
                    idx = i * p + j
                    V_vec[idx] += bc_coeff * yc_coeff
            
            # Build W vector: for each output position
            # The inner Strassen result S_i_idx contributes to certain output positions
            # Each output position (cr, cc) in the 2x2 block result maps to global (2*br+cr, 2*bc+cc)
            # But we need to sum over all block-level terms that contribute to this output
            
            # Actually, W should capture how this rank-1 term contributes to C
            # The inner Strassen term i_idx produces a 2x2 result
            # That 2x2 result is placed at block position (br, bc) in the 4x4 output
            # But wait - the block-level Strassen combines multiple block positions
            
            # Let me reconsider. The block-level Strassen term M_b_idx is a 2x2 matrix.
            # This 2x2 matrix contributes to the 4x4 output C via the output_combos.
            # But which block position of C does M_b_idx correspond to?
            
            # Actually, I need to re-examine. The block-level Strassen gives:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # So M_b_idx is a 2x2 matrix that contributes to specific block positions of C.
            # For example, M1 contributes to C00 with coeff 1 and C11 with coeff 1.
            
            # I need to define for each block-level term b_idx, which output block positions
            # it contributes to and with what coefficient.
            
            pass  # Need to restructure

    # Let me restructure the approach more carefully
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block-level Strassen: for each term b_idx (0-6), define:
    # - Which A blocks and coefficients
    # - Which B blocks and coefficients  
    # - Which C block positions it contributes to and with what coefficient
    
    block_defs = [
        # b_idx=0 (M1): (A00+A11)(B00+B11)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]  # contributes to C00 and C11
        },
        # b_idx=1 (M2): (A10+A11)B00
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)]  # contributes to C10 with +1, C11 with -1
        },
        # b_idx=2 (M3): A00(B10-B11)
        {
            'A': [(0,0,1)],
            'B': [(1,0,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)]  # contributes to C01 with +1, C11 with +1
        },
        # b_idx=3 (M4): A11(B01-B00)
        {
            'A': [(1,1,1)],
            'B': [(0,1,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]  # contributes to C00 with +1, C10 with +1
        },
        # b_idx=4 (M5): (A00+A01)B11
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]  # contributes to C00 with -1, C01 with +1
        },
        # b_idx=5 (M6): (A10-A00)(B00+B01)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]  # contributes to C11 with +1
        },
        # b_idx=6 (M7): (A01-A11)(B10+B11)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]  # contributes to C00 with +1
        },
    ]
    
    # Inner Strassen: for each term i_idx (0-6), define the 2x2 structure
    inner_defs = [
        # i_idx=0 (S1): (X00+X11)(Y00+Y11)
        {
            'X': [(0,0,1), (1,1,1)],
            'Y': [(0,0,1), (1,1,1)],
            'out': [(0,0,1), (1,1,1)]  # S1 contributes to out[0,0] and out[1,1]
        },
        # i_idx=1 (S2): (X10+X11)Y00
        {
            'X': [(1,0,1), (1,1,1)],
            'Y': [(0,0,1)],
            'out': [(1,0,1), (1,1,-1)]  # S2 contributes to out[1,0] with +1, out[1,1] with -1
        },
        # i_idx=2 (S3): X00(Y10-Y11)
        {
            'X': [(0,0,1)],
            'Y': [(1,0,1), (1,1,-1)],
            'out': [(0,1,1), (1,1,1)]  # S3 contributes to out[0,1] with +1, out[1,1] with +1
        },
        # i_idx=3 (S4): X11(Y01-Y00)
        {
            'X': [(1,1,1)],
            'Y': [(0,1,1), (0,0,-1)],
            'out': [(0,0,1), (1,0,1)]  # S4 contributes to out[0,0] with +1, out[1,0] with +1
        },
        # i_idx=4 (S5): (X00+X01)Y11
        {
            'X': [(0,0,1), (0,1,1)],
            'Y': [(1,1,1)],
            'out': [(0,0,-1), (0,1,1)]  # S5 contributes to out[0,0] with -1, out[0,1] with +1
        },
        # i_idx=5 (S6): (X10-X00)(Y00+Y01)
        {
            'X': [(1,0,1), (0,0,-1)],
            'Y': [(0,0,1), (0,1,1)],
            'out': [(1,1,1)]  # S6 contributes to out[1,1] with +1
        },
        # i_idx=6 (S7): (X01-X11)(Y10+Y11)
        {
            'X': [(0,1,1), (1,1,-1)],
            'Y': [(1,0,1), (1,1,1)],
            'out': [(0,0,1)]  # S7 contributes to out[0,0] with +1
        },
    ]
    
    col = 0
    for b_idx in range(7):
        bdef = block_defs[b_idx]
        for i_idx in range(7):
            idef = inner_defs[i_idx]
            
            # Build U vector (size 16)
            U_vec = [0] * (n * m)
            for (abr, abc, ac) in bdef['A']:
                for (xr, xc, xc_c) in idef['X']:
                    i = 2 * abr + xr
                    j = 2 * abc + xc
                    idx = i * m + j
                    U_vec[idx] += ac * xc_c
            
            # Build V vector (size 16)
            V_vec = [0] * (m * p)
            for (bbr, bbc, bc_c) in bdef['B']:
                for (yr, yc, yc_c) in idef['Y']:
                    i = 2 * bbr + yr
                    j = 2 * bbc + yc
                    idx = i * p + j
                    V_vec[idx] += bc_c * yc_c
            
            # Build W vector (size 16)
            # The inner Strassen result is a 2x2 matrix
            # It contributes to C block positions according to bdef['C']
            # Within each C block position, the 2x2 result contributes according to idef['out']
            W_vec = [0] * (p * n)
            for (cbr, cbc, cc_c) in bdef['C']:
                for (orow, ocol, oc_c) in idef['out']:
                    i = 2 * cbr + orow
                    j = 2 * cbc + ocol
                    idx = j * n + i  # W[k*n+i, r] where k is row of C, i is col... 
                    # Wait, let me check the convention: W[k*n+i, r]
                    # k is row index of C (0..3), i is col index of C (0..3)
                    # So idx = k * n + i where k is row, i is col
                    # In our notation: i is row, j is col, so idx = i * n + j
                    idx = i * n + j
                    W_vec[idx] += cc_c * oc_c
            
            # Store in U, V, W
            for r in range(n * m):
                U[r][col] = U_vec[r]
            for r in range(m * p):
                V[r][col] = V_vec[r]
            for r in range(p * n):
                W[r][col] = W_vec[r]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
