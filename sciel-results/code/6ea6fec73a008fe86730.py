"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to matrices
    def add_to_U(row, col, val):
        U[row][col] += val
        
    def add_to_V(row, col, val):
        V[row][col] += val
        
    def add_to_W(row, col, val):
        W[row][col] += val

    # Strassen coefficients for 2x2
    # S_k = sum a_i A_i
    # T_k = sum b_j B_j
    # Indices 0..3 for A blocks (00, 01, 10, 11)
    # Indices 0..3 for B blocks
    
    # Top level: 7 combinations
    # k_top in 0..6
    # L_k_top is a 2x2 matrix formed from A blocks
    # R_k_top is a 2x2 matrix formed from B blocks
    
    # We define the linear forms for L and R at the top level
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00, B01, B10, B11
    
    # Top Level Linear Forms
    # L_k_top[i, j] = sum_{u,v} coeff[u,v] A_{u,v}
    # We store coeffs as a dict or list for each k_top and each entry (i,j) of the 2x2 block
    
    # Let's define the top-level Strassen steps
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # For each k_top, we need the linear combination of A blocks to form L_k_top
    # L_k_top is 2x2.
    # For k_top=0:
    # L0 = A00 + A11 (scalar? No, A00 is 2x2 block. So L0 is 2x2 matrix A00+A11)
    # Wait, in Strassen, the "matrices" being multiplied are the blocks.
    # So L0 is the matrix A00+A11.
    # R0 is the matrix B00+B11.
    
    # So for k_top=0:
    # L0[0,0] = A00[0,0] + A11[0,0]
    # L0[0,1] = A00[0,1] + A11[0,1]
    # ...
    
    # We can define the coefficients for each entry of L_k_top and R_k_top
    # in terms of the global A and B indices.
    
    # Global A index: i in 0..3, j in 0..3.
    # A00 corresponds to i in 0,1; j in 0,1.
    # A01 corresponds to i in 0,1; j in 2,3.
    # A10 corresponds to i in 2,3; j in 0,1.
    # A11 corresponds to i in 2,3; j in 2,3.
    
    # Let's define a function to get the global index for a block (bi, bj) and local (li, lj)
    def global_A_idx(bi, bj, li, lj):
        return (bi * 2 + li) * 4 + (bj * 2 + lj)
        
    def global_B_idx(bi, bj, li, lj):
        return (bi * 2 + li) * 4 + (bj * 2 + lj)

    # Define Top Level Coefficients
    # top_L[k_top][li][lj] = dict of {block_idx: coeff}
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    
    top_L = [None] * 7
    top_R = [None] * 7
    
    # k_top 0: L = A00 + A11, R = B00 + B11
    top_L[0] = [[{0:1, 3:1} for _ in range(2)] for _ in range(2)]
    top_R[0] = [[{0:1, 3:1} for _ in range(2)] for _ in range(2)]
    
    # k_top 1: L = A10 + A11, R = B00
    top_L[1] = [[{2:1, 3:1} for _ in range(2)] for _ in range(2)]
    top_R[1] = [[{0:1} for _ in range(2)] for _ in range(2)]
    
    # k_top 2: L = A00, R = B01 - B11
    top_L[2] = [[{0:1} for _ in range(2)] for _ in range(2)]
    top_R[2] = [[{1:1, 3:-1} for _ in range(2)] for _ in range(2)]
    
    # k_top 3: L = A11, R = B10 - B00
    top_L[3] = [[{3:1} for _ in range(2)] for _ in range(2)]
    top_R[3] = [[{2:1, 0:-1} for _ in range(2)] for _ in range(2)]
    
    # k_top 4: L = A00 + A01, R = B11
    top_L[4] = [[{0:1, 1:1} for _ in range(2)] for _ in range(2)]
    top_R[4] = [[{3:1} for _ in range(2)] for _ in range(2)]
    
    # k_top 5: L = A10 - A00, R = B00 + B01
    top_L[5] = [[{2:1, 0:-1} for _ in range(2)] for _ in range(2)]
    top_R[5] = [[{0:1, 1:1} for _ in range(2)] for _ in range(2)]
    
    # k_top 6: L = A01 - A11, R = B10 + B11
    top_L[6] = [[{1:1, 3:-1} for _ in range(2)] for _ in range(2)]
    top_R[6] = [[{2:1, 3:1} for _ in range(2)] for _ in range(2)]
    
    # Bottom Level Strassen Coefficients
    # For a 2x2 matrix X, the Strassen terms S_k are linear combinations of X's entries.
    # S0 = X00 + X11
    # S1 = X10 + X11
    # S2 = X00
    # S3 = X11
    # S4 = X00 + X01
    # S5 = X10 - X00
    # S6 = X01 - X11
    
    # We need the coefficients for each S_k in terms of local indices (li, lj) of the 2x2 block.
    bot_S = [None] * 7
    bot_T = [None] * 7
    
    # S_k[li][lj] = coeff
    bot_S[0] = [[1, 0], [0, 1]] # X00 + X11
    bot_S[1] = [[0, 0], [1, 1]] # X10 + X11
    bot_S[2] = [[1, 0], [0, 0]] # X00
    bot_S[3] = [[0, 0], [0, 1]] # X11
    bot_S[4] = [[1, 1], [0, 0]] # X00 + X01
    bot_S[5] = [[-1, 0], [1, 0]] # X10 - X00
    bot_S[6] = [[0, 1], [0, -1]] # X01 - X11
    
    bot_T[0] = [[1, 0], [0, 1]] # Y00 + Y11
    bot_T[1] = [[1, 0], [0, 0]] # Y00
    bot_T[2] = [[0, 1], [0, -1]] # Y01 - Y11
    bot_T[3] = [[0, -1], [1, 0]] # Y10 - Y00
    bot_T[4] = [[0, 0], [0, 1]] # Y11
    bot_T[5] = [[1, 1], [0, 0]] # Y00 + Y01
    bot_T[6] = [[0, 1], [1, 1]] # Y10 + Y11
    
    # Contribution of M_k (bottom) to the 2x2 result M_top
    # M_top[0,0] = M0 + M3 - M5 - M6
    # M_top[0,1] = M2 + M5
    # M_top[1,0] = M1 + M4
    # M_top[1,1] = M0 - M1 + M2 + M4
    
    contrib_bot = [
        [[1, 0], [0, 1]], # M0
        [[0, 0], [1, 0]], # M1
        [[0, 1], [0, 0]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [1, 0]], # M4 -- Wait, M1+M4 to M10?
        # Let's re-verify Strassen output
        # C00 = M0 + M3 - M5 - M6
        # C01 = M2 + M5
        # C10 = M1 + M4
        # C11 = M0 - M1 + M2 + M4
        
        # So:
        # M0 contributes to C00 (1), C11 (1)
        # M1 contributes to C10 (1), C11 (-1)
        # M2 contributes to C01 (1), C11 (1)
        # M3 contributes to C00 (1)
        # M4 contributes to C10 (1), C11 (1)
        # M5 contributes to C00 (-1), C01 (1)
        # M6 contributes to C00 (-1)
        
        # Let's build the matrix for each M_k
    ]
    
    # Re-doing contrib_bot properly
    contrib_bot = [
        [[1, 0], [0, 1]], # M0: C00+=1, C11+=1
        [[0, 0], [1, -1]], # M1: C10+=1, C11+=-1
        [[0, 1], [0, 1]], # M2: C01+=1, C11+=1
        [[1, 0], [0, 0]], # M3: C00+=1
        [[0, 0], [1, 1]], # M4: C10+=1, C11+=1
        [[-1, 1], [0, 0]], # M5: C00+=-1, C01+=1
        [[-1, 0], [0, 0]]  # M6: C00+=-1
    ]
    
    # Contribution of M_top (2x2 block) to final 4x4 C
    # C00 (block) = M0 + M3 - M5 - M6
    # C01 (block) = M2 + M5
    # C10 (block) = M1 + M4
    # C11 (block) = M0 - M1 + M2 + M4
    
    # This is the same structure as contrib_bot.
    # So for a given k_top, the contribution to the global C blocks is defined by contrib_bot[k_top].
    # Wait, the indices are:
    # M_top is a 2x2 matrix of blocks? No, M_top is a 2x2 matrix of scalars?
    # No, M_top is a 2x2 matrix.
    # The final C is a 4x4 matrix.
    # C[0:2, 0:2] = M0 + M3 - M5 - M6
    # C[0:2, 2:4] = M2 + M5
    # C[2:4, 0:2] = M1 + M4
    # C[2:4, 2:4] = M0 - M1 + M2 + M4
    
    # So for a specific k_top, the block M_k_top contributes to the global C blocks
    # according to the coefficients in contrib_bot[k_top] (since the formula is identical).
    # Let's call this contrib_top[k_top][bi][bj] where bi, bj are block indices (0,1).
    
    contrib_top = contrib_bot # Same coefficients
    
    # Now iterate over all 49 terms
    col = 0
    for k_top in range(7):
        for k_bot in range(7):
            # 1. Determine U coefficients (A)
            # U is global A index -> coeff
            # Global A index is formed by:
            #   Block index (bi, bj) from top_L[k_top]
            #   Local index (li, lj) from bot_S[k_bot]
            
            # top_L[k_top][li][lj] is a dict of {block_idx: coeff_top}
            # bot_S[k_bot][li][lj] is coeff_bot
            
            # We need to sum over all (bi, bj) and (li, lj)
            # Wait, top_L[k_top] defines the entire 2x2 block L_k_top.
            # L_k_top[li][lj] = sum_{block} coeff_top[block] * A_block[li][lj]
            # Then S_k_bot = sum_{li, lj} coeff_bot[li][lj] * L_k_top[li][lj]
            # S_k_bot = sum_{li, lj} coeff_bot[li][lj] * sum_{block} coeff_top[block] * A_block[li][lj]
            # S_k_bot = sum_{block, li, lj} (coeff_top[block] * coeff_bot[li][lj]) * A_block[li][lj]
            
            # So for each global A index:
            # global_idx = global_A_idx(block, li, lj)
            # coeff_U = coeff_top[block] * coeff_bot[li][lj]
            
            for li in range(2):
                for lj in range(2):
                    coeff_bot_val = bot_S[k_bot][li][lj]
                    if coeff_bot_val == 0: continue
                    
                    # Get top coeffs for this li, lj
                    top_coeffs = top_L[k_top][li][lj]
                    for block_idx, coeff_top_val in top_coeffs.items():
                        if coeff_top_val == 0: continue
                        final_coeff = coeff_top_val * coeff_bot_val
                        g_idx = global_A_idx(block_idx, li, lj)
                        add_to_U(g_idx, col, final_coeff)
                        
            # 2. Determine V coefficients (B)
            for li in range(2):
                for lj in range(2):
                    coeff_bot_val = bot_T[k_bot][li][lj]
                    if coeff_bot_val == 0: continue
                    
                    top_coeffs = top_R[k_top][li][lj]
                    for block_idx, coeff_top_val in top_coeffs.items():
                        if coeff_top_val == 0: continue
                        final_coeff = coeff_top_val * coeff_bot_val
                        g_idx = global_B_idx(block_idx, li, lj)
                        add_to_V(g_idx, col, final_coeff)
                        
            # 3. Determine W coefficients (C)
            # The product S_k_bot * T_k_bot contributes to M_top[ci][cj]
            # where ci, cj are local indices in the 2x2 M_top matrix.
            # contrib_bot[k_bot][ci][cj] gives the coefficient.
            
            # Then M_top[ci][cj] contributes to global C blocks.
            # contrib_top[k_top][bi][bj] gives the coefficient for block (bi, bj).
            # The global C index is global_C_idx(bi, bj, ci, cj)
            
            # Wait, the local indices ci, cj of M_top correspond to the local indices of the blocks in C.
            # M_top is a 2x2 matrix.
            # C is a 4x4 matrix.
            # C[bi*2+ci, bj*2+cj] gets contribution from M_top[ci][cj] * coeff_top[block]
            
            for ci in range(2):
                for cj in range(2):
                    coeff_bot_contrib = contrib_bot[k_bot][ci][cj]
                    if coeff_bot_contrib == 0: continue
                    
                    # Now apply top-level contribution
                    # M_top[ci][cj] is the (ci, cj) entry of the 2x2 matrix M_k_top.
                    # This entry contributes to the global C blocks.
                    # The block index (bi, bj) in C is determined by k_top?
                    # No, k_top determines which M_k_top we are looking at.
                    # The formula C = sum M_k * coeff_k is fixed.
                    # So for a given k_top, the contribution to each block (bi, bj) is
                    # coeff_top[k_top][bi][bj] * M_top[bi][bj] ?
                    # No, M_top is a 2x2 matrix.
                    # The Strassen formula is:
                    # C_block_00 = M0 + M3 - M5 - M6
                    # C_block_01 = M2 + M5
                    # C_block_10 = M1 + M4
                    # C_block_11 = M0 - M1 + M2 + M4
                    #
                    # So for a specific k_top, M_k_top contributes to:
                    # C_block_00 with coeff contrib_top[k_top][0][0]
                    # C_block_01 with coeff contrib_top[k_top][0][1]
                    # C_block_10 with coeff contrib_top[k_top][1][0]
                    # C_block_11 with coeff contrib_top[k_top][1][1]
                    #
                    # Wait, contrib_top[k_top] is a 2x2 matrix.
                    # contrib_top[k_top][bi][bj] is the coefficient for the block (bi, bj) in C
                    # coming from the scalar M_k_top?
                    # No, M_k_top is a 2x2 matrix.
                    # The formula sums the *matrices* M_k_top.
                    # So M_k_top[ci][cj] adds to C_block_bj[ci][cj]?
                    # No, M_k_top is a 2x2 matrix.
                    # C is a 4x4 matrix.
                    # C[0:2, 0:2] = M0 + ...
                    # So M0[0,0] adds to C[0,0].
                    # M0[0,1] adds to C[0,1].
                    # M0[1,0] adds to C[1,0].
                    # M0[1,1] adds to C[1,1].
                    #
                    # So for a given k_top, the entry M_k_top[ci][cj] contributes to
                    # C[bi*2+ci, bj*2+cj] with coefficient contrib_top[k_top][bi][bj].
                    
                    for bi in range(2):
                        for bj in range(2):
                            coeff_top_contrib = contrib_top[k_top][bi][bj]
                            if coeff_top_contrib == 0: continue
                            
                            final_coeff = coeff_bot_contrib * coeff_top_contrib
                            # Global C index
                            # C row: bi*2 + ci
                            # C col: bj*2 + cj
                            g_row = bi * 2 + ci
                            g_col = bj * 2 + cj
                            g_idx = g_row * 4 + g_col
                            add_to_W(g_idx, col, final_coeff)
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
