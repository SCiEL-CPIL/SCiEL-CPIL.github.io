import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses the Kronecker product structure of the 2x2 Strassen decomposition.
    """
    # 2x2 Strassen decomposition parameters
    # U2, V2, W2 are 4x7 matrices (indices 0-3 for 2x2 entries, 0-6 for rank)
    # Indexing for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # U2: Linear forms for A
    # Strassen terms:
    # 0: A00 + A11
    # 1: A10 + A11
    # 2: A00
    # 3: A11
    # 4: A00 + A01
    # 5: A10 - A00
    # 6: A01 - A11
    
    U2 = [
        [1, 0, 0, 1],  # Term 0: A00 + A11
        [0, 0, 1, 1],  # Term 1: A10 + A11
        [1, 0, 0, 0],  # Term 2: A00
        [0, 0, 0, 1],  # Term 3: A11
        [1, 1, 0, 0],  # Term 4: A00 + A01
        [-1, 0, 1, 0], # Term 5: A10 - A00
        [0, 1, 0, -1]  # Term 6: A01 - A11
    ]
    
    # V2: Linear forms for B
    # 0: B00 + B11
    # 1: B00
    # 2: B01 - B11
    # 3: B10 - B00
    # 4: B11
    # 5: B00 + B01
    # 6: B10 + B11
    
    V2 = [
        [1, 0, 0, 1],  # Term 0: B00 + B11
        [1, 0, 0, 0],  # Term 1: B00
        [0, 1, 0, -1], # Term 2: B01 - B11
        [0, 0, 1, -1], # Term 3: B10 - B00
        [0, 0, 0, 1],  # Term 4: B11
        [1, 1, 0, 0],  # Term 5: B00 + B01
        [0, 0, 1, 1]   # Term 6: B10 + B11
    ]
    
    # W2: Contributions to C
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    
    # W2 is 4x7 (rows are C entries, cols are terms)
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Now construct 4x4 decomposition
    # Rank = 7 * 7 = 49
    # Indices for 4x4:
    # A: 16 entries (0-15)
    # B: 16 entries (0-15)
    # C: 16 entries (0-15)
    
    # Mapping from 4x4 index to (block_row, block_col, local_row, local_col)
    # 4x4 index i: row = i // 4, col = i % 4
    # block_row = row // 2, block_col = col // 2
    # local_row = row % 2, local_col = col % 2
    
    def get_2x2_index(br, bc, lr, lc):
        row = 2 * br + lr
        col = 2 * bc + lc
        return row * 4 + col

    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Iterate over the 7 outer Strassen terms (block level)
    for k_outer in range(7):
        # Iterate over the 7 inner Strassen terms (scalar level)
        for k_inner in range(7):
            rank_idx = k_outer * 7 + k_inner
            
            # U construction
            # U[4x4_idx][rank_idx] = sum over block contributions
            # The outer term k_outer defines which blocks are involved in X_block
            # The inner term k_inner defines which entries of X_block are involved
            
            # X_block is a 2x2 matrix formed by summing specific A blocks with coefficients from U2[k_outer]
            # U2 is row-major: U2[k_outer][j] where j is 0..3 (block index in 2x2)
            # Wait, U2[k_outer] is a vector of 4 coefficients for the 4 blocks?
            # No, U2 is defined as linear forms on the 4 entries of a 2x2 matrix.
            # In the context of the outer Strassen step, the "2x2 matrix" is the block matrix A.
            # So U2[k_outer][b] is the coefficient of block A_b in the linear form X_block.
            
            # Similarly, V2[k_outer][b] is the coefficient of block B_b in Y_block.
            
            # Now, X_block is a 2x2 matrix.
            # The inner Strassen step decomposes the product X_block * Y_block.
            # The inner U2[k_inner][l] is the coefficient of entry X_block[l] in the linear form.
            # Here l is 0..3, corresponding to the 4 entries of the 2x2 block X_block.
            
            # So, the total coefficient for A[br, bc, lr, lc] is:
            # coeff_outer = U2[k_outer][block_idx] where block_idx = br*2 + bc
            # coeff_inner = U2[k_inner][local_idx] where local_idx = lr*2 + lc
            
            # Wait, this is only true if the outer linear form is a simple sum of blocks.
            # Yes, Strassen on blocks treats blocks as scalars.
            # So X_block = sum_b U2[k_outer][b] * A_b.
            # Then X_block[lr, lc] = sum_b U2[k_outer][b] * A_b[lr, lc].
            # Then the inner linear form takes sum_l U2[k_inner][l] * X_block[l].
            # So U[4x4_idx] = sum_b sum_l U2[k_outer][b] * U2[k_inner][l] * A_b[l]
            
            # Let's verify this logic.
            # Outer term k_outer:
            # X_block = U2[k_outer] @ A_blocks
            # Y_block = V2[k_outer] @ B_blocks
            
            # Inner term k_inner:
            # Product P = X_block * Y_block
            # P is decomposed into scalar products.
            # The k_inner-th scalar product is:
            # x_form = U2[k_inner] @ X_block_entries
            # y_form = V2[k_inner] @ Y_block_entries
            
            # So x_form = sum_l U2[k_inner][l] * X_block[l]
            #          = sum_l U2[k_inner][l] * (sum_b U2[k_outer][b] * A_block_b[l])
            #          = sum_b sum_l (U2[k_outer][b] * U2[k_inner][l]) * A_block_b[l]
            
            # This is correct.
            
            for b in range(4): # Block index in 2x2 block matrix
                coeff_outer = U2[k_outer][b]
                if coeff_outer == 0: continue
                
                br = b // 2
                bc = b % 2
                
                for l in range(4): # Local index in 2x2 block
                    coeff_inner = U2[k_inner][l]
                    if coeff_inner == 0: continue
                    
                    lr = l // 2
                    lc = l % 2
                    
                    total_coeff = coeff_outer * coeff_inner
                    
                    a_idx = get_2x2_index(br, bc, lr, lc)
                    U[a_idx][rank_idx] += total_coeff
                    
                    b_idx = get_2x2_index(br, bc, lr, lc)
                    V[b_idx][rank_idx] += coeff_outer * V2[k_inner][l] # Wait, V uses V2[k_outer] and V2[k_inner]
            
            # Let's re-calculate V correctly
            # y_form = sum_l V2[k_inner][l] * Y_block[l]
            # Y_block[l] = sum_b V2[k_outer][b] * B_block_b[l]
            # y_form = sum_b sum_l (V2[k_outer][b] * V2[k_inner][l]) * B_block_b[l]
            
            for b in range(4):
                coeff_outer_v = V2[k_outer][b]
                if coeff_outer_v == 0: continue
                
                br = b // 2
                bc = b % 2
                
                for l in range(4):
                    coeff_inner_v = V2[k_inner][l]
                    if coeff_inner_v == 0: continue
                    
                    lr = l // 2
                    lc = l % 2
                    
                    total_coeff_v = coeff_outer_v * coeff_inner_v
                    
                    b_idx = get_2x2_index(br, bc, lr, lc)
                    V[b_idx][rank_idx] += total_coeff_v
            
            # W construction
            # The product P (2x2 matrix) contributes to C.
            # The inner Strassen step says:
            # C_block = sum_k W2[k_inner][c] * P_k (where P_k is the k-th scalar product result)
            # Wait, W2[k_inner][c] is the coefficient of the k_inner-th product in the c-th entry of the 2x2 result.
            
            # The outer Strassen step says:
            # C_block_c = sum_k_outer W2_outer[k_outer][c] * (Outer_Product_k_outer)
            # Where Outer_Product_k_outer is the 2x2 matrix result of X_block * Y_block.
            
            # So the total contribution to C[br, bc, lr, lc] is:
            # sum_k_outer sum_k_inner sum_c (W2_outer[k_outer][c] * W2_inner[k_inner][l_c])
            
            # Here c is the block index in the 2x2 block matrix (0..3).
            # l_c is the local index in the 2x2 block (0..3).
            
            # Let's map:
            # C_block_c is the block at (c_r, c_c).
            # The entry (lr, lc) of C_block_c is at global index (2*c_r + lr, 2*c_c + lc).
            
            for c in range(4): # Block index in C
                coeff_outer_w = W2[k_outer][c]
                if coeff_outer_w == 0: continue
                
                cr = c // 2
                cc = c % 2
                
                for l in range(4): # Local index in C block
                    coeff_inner_w = W2[k_inner][l]
                    if coeff_inner_w == 0: continue
                    
                    lr = l // 2
                    lc = l % 2
                    
                    total_coeff_w = coeff_outer_w * coeff_inner_w
                    
                    c_idx = get_2x2_index(cr, cc, lr, lc)
                    W[c_idx][rank_idx] += total_coeff_w
            
    return {"U": U, "V": V, "W": W}
