"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for matrix multiplication.
    Uses Strassen algorithm for the 4x4x4 part and schoolbook for the 4x4x1 part.
    Total rank: 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS
    # Rank is 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank = 65
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx, v_idx, w_idx are physical indices
    # coeff is the coefficient
    def add_term(u_idx, v_idx, w_idx, col, coeff=1):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # --- Part 1: 4x4 x 4x4 (Strassen) ---
    # Strassen steps 0-6
    # Each step has 7 internal scalar multiplications
    # Total 49 terms
    
    # Define the 7 Strassen block combinations
    # A blocks: 0(TL), 1(TR), 2(BL), 3(BR)
    # B blocks: 0(TL), 1(TR), 2(BL), 3(BR)
    
    # X_k and Y_k are 2x2 matrices formed from A and B blocks
    # We define the linear combination of A blocks for X_k
    # and B blocks for Y_k
    
    # Format: {block_idx: coeff}
    # X_k is sum of coeff * A_block
    # Y_k is sum of coeff * B_block
    
    strassen_defs = [
        # S_0 = (A0+A3)(B0+B3)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # S_1 = (A2+A3)B0
        ({2: 1, 3: 1}, {0: 1}),
        # S_2 = A0(B1-B3)
        ({0: 1}, {1: 1, 3: -1}),
        # S_3 = A3(B2-B0)
        ({3: 1}, {2: 1, 0: -1}),
        # S_4 = (A0+A1)B3
        ({0: 1, 1: 1}, {3: 1}),
        # S_5 = (A2-A0)(B0+B1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # S_6 = (A1-A3)(B2+B3)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Internal Strassen scalar products
    # u=0: (x00+x11)(y00+y11)
    # u=1: (x10+x11)y00
    # u=2: x00(y01-y11)
    # u=3: x11(y10-y00)
    # u=4: (x00+x01)y11
    # u=5: (x10-x00)(y00+y01)
    # u=6: (x01-x11)(y10+y11)
    
    # Map internal step u to (x_indices, y_indices, coeffs_x, coeffs_y)
    # x_indices are (row, col) in the 2x2 block matrix
    # We need to expand these to physical indices
    
    # Physical index for A block b (0..3), element (r,c) in 2x2 block:
    # Block 0 (TL): i in 0..1, j in 0..1 -> phys i, j
    # Block 1 (TR): i in 0..1, j in 2..3 -> phys i, j+2
    # Block 2 (BL): i in 2..3, j in 0..1 -> phys i+2, j
    # Block 3 (BR): i in 2..3, j in 2..3 -> phys i+2, j+2
    
    def get_a_phys(block, r, c):
        # r, c in 0..1
        if block == 0:
            return r * 4 + c
        elif block == 1:
            return r * 4 + (c + 2)
        elif block == 2:
            return (r + 2) * 4 + c
        else: # block 3
            return (r + 2) * 4 + (c + 2)

    def get_b_phys_left(block, r, c):
        # B is 4x4 part (cols 0..3)
        # r is row in 4x4 (0..3), c is col in 4x4 (0..3)
        # But here block is 2x2, so r,c in 0..1
        # Physical V index: row (0..3) * 5 + col (0..3)
        if block == 0:
            return r * 5 + c
        elif block == 1:
            return r * 5 + (c + 2)
        elif block == 2:
            return (r + 2) * 5 + c
        else: # block 3
            return (r + 2) * 5 + (c + 2)

    def get_w_phys_left(block, r, c):
        # W is 5x4 (cols 0..3 part)
        # k is col in 4x4 (0..3), i is row in 4x4 (0..3)
        # Physical W index: k * 4 + i
        # r is row of result (0..3), c is col of result (0..3)
        # In Strassen, result block b corresponds to:
        # Block 0 (TL): i 0..1, k 0..1
        # Block 1 (TR): i 0..1, k 2..3
        # Block 2 (BL): i 2..3, k 0..1
        # Block 3 (BR): i 2..3, k 2..3
        if block == 0:
            return c * 4 + r # k=c, i=r
        elif block == 1:
            return (c + 2) * 4 + r
        elif block == 2:
            return c * 4 + (r + 2)
        else: # block 3
            return (c + 2) * 4 + (r + 2)

    # Define the 7 internal Strassen terms
    # Each term is a list of (x_block, x_r, x_c, coeff_x, y_block, y_r, y_c, coeff_y)
    # But x_block and y_block are determined by the outer Strassen step s.
    # So for a given s, X is a combination of A blocks, Y is a combination of B blocks.
    # We need to expand X and Y into their elements.
    
    # For each s, we have X_def and Y_def.
    # X_def is {block: coeff}.
    # The element X[r][c] is sum_{b} coeff_b * A_b[r][c].
    # Similarly for Y.
    
    # For internal step u, we identify which X elements and Y elements are multiplied.
    # e.g., u=0: (X[0][0] + X[1][1]) * (Y[0][0] + Y[1][1])
    # This expands to 4 scalar multiplications:
    # X[0][0]Y[0][0] + X[0][0]Y[1][1] + X[1][1]Y[0][0] + X[1][1]Y[1][1]
    # But wait, the rank is 7, not 16.
    # The Strassen algorithm groups these.
    # The 7 products ARE the scalar multiplications.
    # So for u=0, the product is P_0 = (X[0][0] + X[1][1]) * (Y[0][0] + Y[1][1]).
    # This is ONE scalar multiplication in the tensor decomposition?
    # No. The tensor decomposition terms are linear forms.
    # The term is $L_A \cdot L_B \cdot M_C$.
    # $L_A = X[0][0] + X[1][1]$.
    # $L_B = Y[0][0] + Y[1][1]$.
    # So we add one term to U, V, W.
    # U gets the coefficients of $L_A$.
    # V gets the coefficients of $L_B$.
    # W gets the coefficient of the output.
    
    # So for each s and u, we have ONE term in the decomposition.
    # Total 7 * 7 = 49 terms.
    
    # We need to define the linear forms for X and Y for each u.
    
    # u=0: Lx = X00 + X11, Ly = Y00 + Y11
    # u=1: Lx = X10 + X11, Ly = Y00
    # u=2: Lx = X00, Ly = Y01 - Y11
    # u=3: Lx = X11, Ly = Y10 - Y00
    # u=4: Lx = X00 + X01, Ly = Y11
    # u=5: Lx = X10 - X00, Ly = Y00 + Y01
    # u=6: Lx = X01 - X11, Ly = Y10 + Y11
    
    # We need to map X[r][c] to physical A indices.
    # X[r][c] = sum_{b} X_def[b] * A_b[r][c]
    # So Lx = sum_{r,c} coeff_{r,c} * X[r][c]
    # = sum_{r,c} coeff_{r,c} * sum_{b} X_def[b] * A_b[r][c]
    # = sum_{b} X_def[b] * sum_{r,c} coeff_{r,c} * A_b[r][c]
    
    # So for a given u, we have coeffs for X elements: cx[r][c]
    # And coeffs for Y elements: cy[r][c]
    
    internal_strassen = [
        # u, cx (2x2), cy (2x2)
        (0, [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        (1, [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
        (2, [[1, 0], [0, 0]], [[0, 1], [0, -1]]),
        (3, [[0, 0], [0, 1]], [[0, -1], [1, 0]]),
        (4, [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        (5, [[-1, 0], [1, 0]], [[1, 1], [0, 0]]),
        (6, [[0, 1], [0, -1]], [[0, 1], [1, 0]]),
    ]
    
    # Output combination for W
    # S_k is a 2x2 matrix.
    # S_k[0][0] = P0 + P3 - P4 + P6
    # S_k[0][1] = P2 + P4
    # S_k[1][0] = P1 + P3
    # S_k[1][1] = P0 - P1 + P2 + P5
    
    # For each term (s, u), it contributes to specific W elements.
    # We need to add the contribution to W.
    # The term corresponds to P_u of step s.
    # P_u is a scalar.
    # It contributes to S_s[0][0], S_s[0][1], etc.
    # S_s is a block of the final C matrix.
    # So we need to map S_s[r][c] to physical W indices.
    
    # Let's precompute the W contribution for each (s, u).
    # w_coeffs[s][u] is a dict { (r_block, c_block): coeff }
    # where r_block, c_block in 0..1 (position within the 2x2 block result of S_s)
    # And S_s itself is a block of the 4x4 result.
    # Wait, S_s is the result of the Strassen step.
    # The 7 Strassen steps produce 7 "virtual" 2x2 matrices.
    # The final C is a linear combination of these 7 virtual matrices.
    # C = sum_{s} coeff_{s, block_r, block_c} * S_s[block_r][block_c]
    
    # Actually, the Strassen algorithm gives:
    # C_00 = S_0[0][0] + S_3[0][0] - S_4[0][0] + S_6[0][0]  (Wait, indices might be off)
    # Let's stick to the standard Strassen output formulas.
    # C_TL = S_0[0][0] + S_3[0][0] - S_4[0][0] + S_6[0][0] ? No.
    # The formulas I wrote earlier were for the elements of the 2x2 result Z of a single Strassen step.
    # Here, the "result" of the 4x4 Strassen is the 4x4 matrix C.
    # C is composed of 4 blocks: C_TL, C_TR, C_BL, C_BR.
    # C_TL = S_0[0][0] + S_3[0][0] - S_4[0][0] + S_6[0][0]
    # C_TR = S_2[0][0] + S_4[0][0]
    # C_BL = S_1[0][0] + S_3[0][0]
    # C_BR = S_0[0][0] - S_1[0][0] + S_2[0][0] + S_5[0][0]
    
    # Wait, S_k are 2x2 matrices.
    # So C_TL is a 2x2 matrix.
    # C_TL = S_0[0][0] + ... where S_0[0][0] is the top-left element of the 2x2 matrix S_0.
    # So yes, the formula applies element-wise.
    
    # So for each term (s, u), which corresponds to P_u in step s,
    # it contributes to S_s[0][0], S_s[0][1], S_s[1][0], S_s[1][1].
    # Let's define the contribution of P_u to the 4 elements of S_s.
    
    p_to_s_elem = [
        # u, [coeff for S_s[0][0], S_s[0][1], S_s[1][0], S_s[1][1]]
        (0, [1, 0, 0, 1]),   # P0: S00=1, S01=0, S10=0, S11=1
        (1, [0, 0, 1, -1]),  # P1: S00=0, S01=0, S10=1, S11=-1
        (2, [0, 1, 0, 1]),   # P2: S00=0, S01=1, S10=0, S11=1
        (3, [1, 0, 1, 0]),   # P3: S00=1, S01=0, S10=1, S11=0
        (4, [-1, 1, 0, 0]),  # P4: S00=-1, S01=1, S10=0, S11=0
        (5, [0, 0, 0, 1]),   # P5: S00=0, S01=0, S10=0, S11=1
        (6, [1, 0, 0, 0]),   # P6: S00=1, S01=0, S10=0, S11=0
    ]
    
    # And the contribution of S_s[r][c] to the final C blocks.
    # C_TL (block 0) gets S_0[0][0], S_3[0][0], -S_4[0][0], S_6[0][0]
    # C_TR (block 1) gets S_2[0][0], S_4[0][0]
    # C_BL (block 2) gets S_1[0][0], S_3[0][0]
    # C_BR (block 3) gets S_0[0][0], -S_1[0][0], S_2[0][0], S_5[0][0]
    
    # Note: This is for the top-left element [0][0] of S_s.
    # The other elements [0][1], [1][0], [1][1] follow the same pattern for the other parts of the 2x2 block.
    # So for S_s[r][c], it contributes to C_block_r_c[r][c] with the same coefficients.
    
    s_to_c_block = [
        # s, {block_idx: coeff}
        (0, {0: 1, 3: 1}), # S0 contributes to C_TL and C_BR
        (1, {2: 1, 3: -1}), # S1 contributes to C_BL and C_BR
        (2, {1: 1, 3: 1}), # S2 contributes to C_TR and C_BR
        (3, {0: 1, 2: 1}), # S3 contributes to C_TL and C_BL
        (4, {0: -1, 1: 1}), # S4 contributes to C_TL and C_TR
        (5, {3: 1}), # S5 contributes to C_BR
        (6, {0: 1}), # S6 contributes to C_TL
    ]
    
    col = 0
    for s in range(7):
        x_def, y_def = strassen_defs[s]
        for u in range(7):
            # Get cx, cy for this u
            _, cx, cy = internal_strassen[u]
            
            # Build U coefficients
            # U index: i*4+j
            # Lx = sum_{r,c} cx[r][c] * X[r][c]
            # X[r][c] = sum_{b} x_def[b] * A_b[r][c]
            # So coeff for A_b[r][c] is cx[r][c] * x_def[b]
            
            for b in x_def:
                coeff_b = x_def[b]
                for r in range(2):
                    for c in range(2):
                        coeff_rc = cx[r][c]
                        if coeff_rc != 0:
                            total_coeff = coeff_b * coeff_rc
                            a_phys = get_a_phys(b, r, c)
                            U[a_phys][col] += total_coeff
            
            # Build V coefficients
            # V index: j*5+k (k in 0..3)
            # Ly = sum_{r,c} cy[r][c] * Y[r][c]
            # Y[r][c] = sum_{b} y_def[b] * B_b[r][c]
            for b in y_def:
                coeff_b = y_def[b]
                for r in range(2):
                    for c in range(2):
                        coeff_rc = cy[r][c]
                        if coeff_rc != 0:
                            total_coeff = coeff_b * coeff_rc
                            b_phys = get_b_phys_left(b, r, c)
                            V[b_phys][col] += total_coeff
            
            # Build W coefficients
            # W index: k*4+i
            # P_u contributes to S_s[0][0], S_s[0][1], S_s[1][0], S_s[1][1]
            # Let coeffs_s = p_to_s_elem[u]
            # coeffs_s[0] for S_s[0][0]
            # coeffs_s[1] for S_s[0][1]
            # coeffs_s[2] for S_s[1][0]
            # coeffs_s[3] for S_s[1][1]
            
            _, p_coeffs = p_to_s_elem[u]
            # p_coeffs is [c00, c01, c10, c11]
            
            # S_s[r][c] contributes to C blocks defined in s_to_c_block[s]
            _, c_block_map = s_to_c_block[s]
            
            for r in range(2):
                for c in range(2):
                    # Get coeff for S_s[r][c]
                    # Map (r,c) to index in p_coeffs
                    idx = r * 2 + c
                    coeff_s = p_coeffs[idx]
                    if coeff_s == 0:
                        continue
                    
                    # This S_s[r][c] contributes to C blocks in c_block_map
                    for block_idx, coeff_block in c_block_map.items():
                        total_coeff = coeff_s * coeff_block
                        # C block block_idx has elements at (i, k)
                        # i = (block_idx // 2) * 2 + r
                        # k = (block_idx % 2) * 2 + c
                        i = (block_idx // 2) * 2 + r
                        k = (block_idx % 2) * 2 + c
                        w_phys = get_w_phys_left(block_idx, r, c) # Wait, get_w_phys_left takes block, r, c
                        # My get_w_phys_left uses r as row, c as col of the 2x2 block.
                        # And block_idx determines the offset.
                        # So this is correct.
                        W[w_phys][col] += total_coeff
            
            col += 1
    
    # --- Part 2: 4x4 x 4x1 (Schoolbook) ---
    # 16 terms
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i # k=4, i=i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
