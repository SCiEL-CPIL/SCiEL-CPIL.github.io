"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140

    # Strassen 2x2x2 rank-7 decomposition
    # U, V, W are 4x7 matrices
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    # Part 1: 4x4x4 multiplication (first 4 columns of output)
    # Decompose into 7 Strassen products of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 + A22) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # We need to define the 2x2 matrix multiplications for each Mk.
    # Each Mk is a 2x2 matrix product.
    # We will create a helper to add a 2x2x2 Strassen decomposition to the global matrices.
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    def add_strassen_block(i_offset, j_offset, k_offset, u_scale, v_scale, w_scale):
        """
        Adds a 2x2x2 Strassen block.
        i_offset, j_offset, k_offset: top-left indices in the 4x4, 4x4, 4x4 (or 4x2) grid.
        u_scale, v_scale, w_scale: scaling factors for the U, V, W matrices.
        """
        nonlocal total_rank, U, V, W
        
        # Initialize new rows/cols if needed
        # We are building the full matrices for the 4x4x6 problem.
        # U is 16xR, V is 24xR, W is 24xR (p*n = 6*4=24)
        
        pass

    # Instead of dynamic appending, let's pre-allocate the max rank we expect (77) 
    # and fill it.
    
    # Max rank estimate: 49 (for 4x4x4) + 28 (for 4x4x2) = 77.
    # Let's use 80 to be safe.
    max_rank = 80
    U = [[0] * max_rank for _ in range(n * m)] # 16 x 80
    V = [[0] * max_rank for _ in range(m * p)] # 24 x 80
    W = [[0] * max_rank for _ in range(p * n)] # 24 x 80
    
    current_rank = 0
    
    def add_2x2x2_block(i0, j0, k0, u_sign, v_sign, w_sign):
        """
        Adds a 2x2x2 Strassen block starting at indices i0, j0, k0.
        i0, j0 are in the 4x4 A and B grids.
        k0 is in the 4x6 output grid.
        """
        nonlocal current_rank
        for c in range(7):
            col = current_rank + c
            for di in range(2):
                for dj in range(2):
                    i = i0 + di
                    j = j0 + dj
                    U[i * m + j][col] += S_U[di * 2 + dj][c] * u_sign
            for dj in range(2):
                for dk in range(2):
                    j = j0 + dj
                    k = k0 + dk
                    V[j * p + k][col] += S_V[dj * 2 + dk][c] * v_sign
            for dk in range(2):
                for di in range(2):
                    k = k0 + dk
                    i = i0 + di
                    W[k * n + i][col] += S_W[dk * 2 + di][c] * w_sign
        current_rank += 7

    # --- Part 1: 4x4x4 (k=0..3) ---
    # M1 = A11 * (B11 + B22)
    # A11 is at (0,0), size 2x2.
    # (B11 + B22) is a 2x2 matrix.
    # B11 is at (0,0) in B, B22 is at (2,2) in B.
    # We need to form the sum B11+B22.
    # In the tensor decomposition, we can't just "add" matrices in V directly if we are using fixed Strassen blocks.
    # We must expand the Strassen formula for the 4x4 case.
    
    # The 4x4 Strassen algorithm uses 7 products of 2x2 matrices.
    # Let's denote the 2x2 blocks of A as A00, A01, A10, A11.
    # Let's denote the 2x2 blocks of B as B00, B01, B10, B11.
    
    # Product 1: P1 = A00 * (B00 + B11)
    # Product 2: P2 = (A00 + A01) * B11
    # Product 3: P3 = (A10 + A11) * B00
    # Product 4: P4 = A11 * (B10 - B00)
    # Product 5: P5 = (A00 + A11) * (B00 + B11)
    # Product 6: P6 = (A01 - A11) * (B10 + B11)
    # Product 7: P7 = (A00 - A10) * (B00 + B01)
    
    # Each Pi is a 2x2 matrix product.
    # We need to decompose each Pi into rank 7.
    # The inputs to the Pi are linear combinations of the 2x2 blocks.
    # This means we need to "pre-process" the U and V matrices to form these sums.
    
    # Instead of adding blocks, we will construct the U, V, W matrices for the 4x4x4 part
    # by iterating over the 7 products and expanding the Strassen 2x2 decomposition for each.
    
    # Helper to add a 2x2x2 block with specific input/output scaling.
    # This is complex because the inputs are sums of blocks.
    # Example: P1 = A00 * (B00 + B11).
    # The "A" part is A00 (rows 0-1, cols 0-1 of A).
    # The "B" part is B00 + B11 (rows 0-1, cols 0-1 of B PLUS rows 2-3, cols 2-3 of B).
    # In the V matrix, the entries for B00 and B11 will be added.
    
    # Let's define the 7 products and their components.
    
    products = [
        # (A_part, B_part)
        # A_part is a list of (i_offset, j_offset, sign) for 2x2 blocks in A
        # B_part is a list of (i_offset, j_offset, sign) for 2x2 blocks in B
        (
            [(0, 0, 1)],
            [(0, 0, 1), (2, 2, 1)]
        ), # P1
        (
            [(0, 0, 1), (0, 2, 1)],
            [(2, 2, 1)]
        ), # P2: (A00+A01)*B11
        (
            [(2, 0, 1), (2, 2, 1)],
            [(0, 0, 1)]
        ), # P3: (A10+A11)*B00
        (
            [(2, 2, 1)],
            [(2, 0, 1), (0, 0, -1)]
        ), # P4: A11*(B10-B00)
        (
            [(0, 0, 1), (2, 2, 1)],
            [(0, 0, 1), (2, 2, 1)]
        ), # P5: (A00+A11)*(B00+B11)
        (
            [(0, 2, 1), (2, 2, -1)],
            [(2, 0, 1), (2, 2, 1)]
        ), # P6: (A01-A11)*(B10+B11)
        (
            [(0, 0, 1), (2, 0, -1)],
            [(0, 0, 1), (0, 2, 1)]
        ), # P7: (A00-A10)*(B00+B01)
    ]
    
    # For each product, we add 7 rank-1 terms (from Strassen).
    # The U matrix will be the sum of contributions from the A_part blocks.
    # The V matrix will be the sum of contributions from the B_part blocks.
    # The W matrix is the standard Strassen W for the 2x2 output.
    
    # Note: The output of P1 is a 2x2 matrix. It contributes to C00, C01, C10, C11.
    # The output of P2 contributes to C01, C11.
    # We need to add the W contributions with the correct signs and positions.
    
    # Strassen Output Reconstruction:
    # C00 = P1 - P3 + P4 - P7
    # C01 = P2 + P7
    # C10 = P3 + P6
    # C11 = P1 - P2 + P5 + P6
    
    # We will iterate through the 7 products, add the U and V parts, 
    # and accumulate the W parts with the appropriate coefficients.
    
    # Initialize W_accumulator for the 4x4x4 part (k=0..3)
    # W is 24xR. We are filling columns 0..48.
    
    for prod_idx in range(7):
        a_parts = products[prod_idx][0]
        b_parts = products[prod_idx][1]
        
        # Add U and V for this product
        # We need to add the Strassen U and V for each component.
        # The Strassen U and V are for a 2x2 matrix.
        # We will add them to the global U and V matrices.
        
        for c in range(7):
            col = current_rank + c
            # Add U contributions
            for (ai, aj, sign) in a_parts:
                for di in range(2):
                    for dj in range(2):
                        i = ai + di
                        j = aj + dj
                        U[i * m + j][col] += S_U[di * 2 + dj][c] * sign
                        
            # Add V contributions
            for (bi, bj, sign) in b_parts:
                for dj in range(2):
                    for dk in range(2):
                        j = bi + dj
                        k = bj + dk
                        V[j * p + k][col] += S_V[dj * 2 + dk][c] * sign
                        
            # Add W contributions (reconstruction)
            # We need to add the Strassen W with the coefficient for this product.
            # The coefficient depends on which C_ij this product contributes to.
            
            # We will handle W accumulation after all U/V are set, or we can add it now.
            # Let's add it now to a temporary structure or directly to W.
            # But W depends on the output position (i,k).
            # The Strassen W for a 2x2 product gives the 2x2 output.
            # We need to map this 2x2 output to the 4x4 output C.
            
            # Let's store the W contribution for this product in a 2x2 matrix, 
            # then add it to the global W with the correct signs and offsets.
            
            # Actually, it's easier to just add the W contributions directly to the global W matrix
            # using the reconstruction formula.
            
            pass # We will do W accumulation in a separate step to avoid confusion.
            
        current_rank += 7
        
    # Now we have U and V for the 7 products.
    # We need to fill W.
    # The W matrix for the 4x4x4 part is the sum of the W contributions from the 7 products,
    # scaled by the reconstruction coefficients.
    
    # Reconstruction coefficients for P1..P7:
    # C00: P1(+), P3(-), P4(+), P7(-)
    # C01: P2(+), P7(+)
    # C10: P3(+), P6(+)
    # C11: P1(-), P2(+), P5(+), P6(+)
    
    # The Strassen W matrix S_W maps the 7 products to the 2x2 output.
    # S_W[dk, di] is the coefficient for product c to contribute to output (di, dk).
    # Wait, S_W is 4x7. S_W[idx, c] where idx = dk*2 + di.
    # So S_W[0, c] is coeff for (0,0), S_W[1, c] for (1,0), etc.
    
    # We need to add S_W[local_idx, c] * global_coeff to W[global_idx, col].
    
    # Let's define the global offsets for the 4 2x2 output blocks.
    # C00 is at (0,0), C01 at (0,2), C10 at (2,0), C11 at (2,2).
    
    output_blocks = [
        (0, 0), # C00
        (0, 2), # C01
        (2, 0), # C10
        (2, 2), # C11
    ]
    
    # Coefficients for each product (0..6) for each output block (0..3)
    # Product 0 (P1): C00(+1), C11(-1)
    # Product 1 (P2): C01(+1), C11(+1)
    # Product 2 (P3): C00(-1), C10(+1)
    # Product 3 (P4): C00(+1)
    # Product 4 (P5): C11(+1)
    # Product 5 (P6): C10(+1), C11(+1)
    # Product 6 (P7): C00(-1), C01(+1)
    
    coeffs = [
        {0: 1, 3: -1}, # P1
        {1: 1, 3: 1},  # P2
        {0: -1, 2: 1}, # P3
        {0: 1},        # P4
        {3: 1},        # P5
        {2: 1, 3: 1},  # P6
        {0: -1, 1: 1}, # P7
    ]
    
    for prod_idx in range(7):
        col_base = prod_idx * 7
        for c in range(7):
            col = col_base + c
            for block_idx, coeff in coeffs[prod_idx].items():
                oi, oj = output_blocks[block_idx]
                # Add S_W contributions to W
                for di in range(2):
                    for dk in range(2):
                        local_w_idx = dk * 2 + di
                        w_val = S_W[local_w_idx][c] * coeff
                        if w_val != 0:
                            gi = oi + di
                            gk = oj + dk
                            W[gk * n + gi][col] += w_val

    # --- Part 2: 4x4x2 (k=4..5) ---
    # C_top (rows 0-1) = A00 * B_top + A01 * B_bot
    # C_bot (rows 2-3) = A10 * B_top + A11 * B_bot
    # B_top is rows 0-1 of B, B_bot is rows 2-3 of B.
    # Output k=4,5 correspond to columns 0,1 of the 4x2 output.
    
    # We have 4 products:
    # Q1 = A00 * B_top
    # Q2 = A01 * B_bot
    # Q3 = A10 * B_top
    # Q4 = A11 * B_bot
    
    # Each is a 2x2x2 multiplication.
    # Q1 output goes to C[0:2, 4:6]
    # Q2 output goes to C[0:2, 4:6]
    # Q3 output goes to C[2:4, 4:6]
    # Q4 output goes to C[2:4, 4:6]
    
    # We need to add these to the global U, V, W.
    # U and V are already filled for the first 49 ranks.
    # We will start filling from current_rank (49).
    
    # Q1: A00 * B_top
    # A00 is at (0,0). B_top is at (0,0) in B (rows 0-1, cols 0-1? No, B is 4x6. B_top is rows 0-1, cols 0-1? No.
    # B is 4x6. The last 2 columns are k=4,5.
    # So B_top is rows 0-1, cols 4-5.
    # B_bot is rows 2-3, cols 4-5.
    
    # Let's define the 4 products for the 4x4x2 part.
    
    part2_products = [
        # (A_part, B_part, Output_Offset_i, Output_Offset_k)
        ((0, 0), (0, 4), 0, 4), # Q1: A00 * B_top -> C[0:2, 4:6]
        ((0, 2), (2, 4), 0, 4), # Q2: A01 * B_bot -> C[0:2, 4:6]
        ((2, 0), (0, 4), 2, 4), # Q3: A10 * B_top -> C[2:4, 4:6]
        ((2, 2), (2, 4), 2, 4), # Q4: A11 * B_bot -> C[2:4, 4:6]
    ]
    
    for a_off, b_off, out_i, out_k in part2_products:
        # Add U
        for c in range(7):
            col = current_rank + c
            for di in range(2):
                for dj in range(2):
                    i = a_off[0] + di
                    j = a_off[1] + dj
                    U[i * m + j][col] += S_U[di * 2 + dj][c]
                    
            # Add V
            for dj in range(2):
                for dk in range(2):
                    j = b_off[0] + dj
                    k = b_off[1] + dk
                    V[j * p + k][col] += S_V[dj * 2 + dk][c]
                    
            # Add W
            for di in range(2):
                for dk in range(2):
                    i = out_i + di
                    k = out_k + dk
                    W[k * n + i][col] += S_W[dk * 2 + di][c]
                    
        current_rank += 7

    # Total rank is current_rank (49 + 28 = 77)
    
    # Trim matrices to actual rank
    U_final = [row[:current_rank] for row in U]
    V_final = [row[:current_rank] for row in V]
    W_final = [row[:current_rank] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
