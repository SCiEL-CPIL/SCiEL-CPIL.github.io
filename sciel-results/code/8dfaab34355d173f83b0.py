def expand_strassen_2x2_to_4x4(U2, V2, W2):
    # U2, V2, W2 are 4x7
    # Returns U, V, W of size 16x49, 16x49, 16x49
    # Wait, V is for B, which is 4x4, so 16 elements.
    # W is for C, which is 4x4, so 16 elements.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Block mapping for A, B, C
    # Block 0: top-left (0,0)
    # Block 1: top-right (0,1)
    # Block 2: bottom-left (1,0)
    # Block 3: bottom-right (1,1)
    
    # Flat indices for each block's 4 elements
    # Block (br, bc) where br, bc in {0,1}
    # Element (er, ec) in block
    # Global row = br*2 + er
    # Global col = bc*2 + ec
    # Flat index = global_row*4 + global_col
    
    def get_flat_indices(br, bc):
        idx = []
        for er in range(2):
            for ec in range(2):
                gr = br*2 + er
                gc = bc*2 + ec
                idx.append(gr*4 + gc)
        return idx
    
    # Strassen steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each step k (0..6), we have a 2x2 matrix product.
    # We use U2, V2, W2 for that product.
    # But we need to apply it to the specific blocks.
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11
    # B part: B00 + B11
    
    # We can't just use U2 directly because U2 assumes a single 2x2 matrix.
    # Here the "input" to the 2x2 multiplication is a sum of blocks.
    # So for M1, the A-input is A00 + A11.
    # So U for M1 should have coefficients for A00 and A11.
    # Specifically, for each element in A00, we add the coefficient from U2.
    # For each element in A11, we add the coefficient from U2.
    
    # Let's define the block contributions for each Strassen step.
    # Step k:
    # A_blocks: list of (block_id, sign)
    # B_blocks: list of (block_id, sign)
    
    steps = [
        ([(0,1), (3,1)], [(0,1), (3,1)]), # M1
        ([(2,1), (3,1)], [(0,1)]),        # M2
        ([(0,1)], [(1,1), (3,-1)]),      # M3
        ([(3,1)], [(2,1), (0,-1)]),      # M4
        ([(0,1), (1,1)], [(3,1)]),       # M5
        ([(2,1), (0,-1)], [(0,1), (1,1)]), # M6
        ([(1,1), (3,-1)], [(2,1), (3,1)])  # M7
    ]
    
    for k, (a_blocks, b_blocks) in enumerate(steps):
        # For each of the 7 scalar multiplications in the 2x2 Strassen
        for j in range(7):
            col = k*7 + j
            
            # Update U
            for br, sign in a_blocks:
                idxs = get_flat_indices(br//2, br%2) # Wait, block_id 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                # My block_id mapping:
                # 0: (0,0)
                # 1: (0,1)
                # 2: (1,0)
                # 3: (1,1)
                br_idx, bc_idx = divmod(br, 2)
                idxs = get_flat_indices(br_idx, bc_idx)
                # U2[j] has 4 values, for elements 0,1,2,3 of the block
                for r in range(4):
                    val = U2[r][j] * sign
                    if val != 0:
                        U[idxs[r]][col] += val
                        
            # Update V
            for br, sign in b_blocks:
                br_idx, bc_idx = divmod(br, 2)
                idxs = get_flat_indices(br_idx, bc_idx)
                for r in range(4):
                    val = V2[r][j] * sign
                    if val != 0:
                        V[idxs[r]][col] += val
                        
            # Update W
            # W for the result C block.
            # The result of M1 is added to C00, C11, etc.
            # In Strassen, the 7 products M1..M7 are combined to form C blocks.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for each step k, the contribution to W is determined by which C blocks M_k contributes to.
            # M1 contributes to C00 (+1), C11 (+1)
            # M2 contributes to C10 (+1), C11 (-1)
            # M3 contributes to C01 (+1), C11 (+1)
            # M4 contributes to C00 (+1), C10 (+1)
            # M5 contributes to C00 (-1), C01 (+1)
            # M6 contributes to C11 (+1)
            # M7 contributes to C00 (+1)
            
            c_contributions = [
                [(0,1), (3,1)], # M1
                [(2,1), (3,-1)], # M2
                [(1,1), (3,1)], # M3
                [(0,1), (2,1)], # M4
                [(0,-1), (1,1)], # M5
                [(3,1)], # M6
                [(0,1)] # M7
            ]
            
            for br, sign in c_contributions[k]:
                br_idx, bc_idx = divmod(br, 2)
                idxs = get_flat_indices(br_idx, bc_idx)
                for r in range(4):
                    val = W2[r][j] * sign
                    if val != 0:
                        W[idxs[r]][col] += val
                        
    return U, V, W
