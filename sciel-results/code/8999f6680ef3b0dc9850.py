import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the problem into two parts:
    # 1. A (4x4) * B1 (4x4) -> C1 (4x4)
    # 2. A (4x4) * B2 (4x3) -> C2 (4x3)
    
    # Part 1: 4x4x4 using recursive Strassen (Rank 49)
    # Part 2: 4x4x3 using block Strassen (Rank 44)
    
    # We will build the U, V, W matrices for each part and then combine them.
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # Indices:
        # U: 4x7 (A is 2x2, flat 4)
        # V: 4x7 (B is 2x2, flat 4)
        # W: 4x7 (C is 2x2, flat 4)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 += M2
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 += M3
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1 # C00 -= M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00 += M7
        
        return U, V, W

    def strassen_4x4x4():
        # Returns U, V, W for 4x4x4
        # U: 16x49
        # V: 16x49
        # W: 16x49
        
        U_2, V_2, W_2 = strassen_2x2x2()
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # The 7 multiplications M1..M7 are 2x2x2 multiplications.
        # Each has rank 7.
        # So we have 7 * 7 = 49 terms.
        
        # For each of the 7 "meta" multiplications, we embed the 2x2x2 decomposition.
        # The "meta" multiplication corresponds to a specific block of A and B.
        
        # We need to map the 2x2 blocks of A (4x4) to the 2x2 inputs of the meta-strassen.
        # A is 4x4. Blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1).
        # In the meta-strassen, we use combinations like A00+A11.
        # These combinations are 2x2 matrices.
        # So the input to the 2x2x2 decomposition is a 2x2 matrix which is a linear combo of blocks of A.
        
        # Let's define the 7 meta-multiplications.
        # M1: (A00+A11) * (B00+B11)
        # M2: (A10+A11) * B00
        # M3: A00 * (B01-B11)
        # M4: A11 * (B10-B00)
        # M5: (A00+A01) * B11
        # M6: (A10-A00) * (B00+B01)
        # M7: (A01-A11) * (B10+B11)
        
        # For each meta-multiplication, we need to specify:
        # 1. Which 2x2 blocks of A are involved and with what coefficients.
        # 2. Which 2x2 blocks of B are involved and with what coefficients.
        # 3. Where the resulting 2x2 block of C goes.
        
        # Since the 2x2x2 decomposition is linear, we can just add the contributions.
        
        # Block indices for 4x4 matrix (flattened 16):
        # Block (r,c) of 2x2 blocks corresponds to rows 2r..2r+1, cols 2c..2c+1.
        # Flat index for element (i,j) in 4x4 is i*4+j.
        # Block (r,c) elements are (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1).
        # Flat indices: 2r*4+2c, 2r*4+2c+1, (2r+1)*4+2c, (2r+1)*4+2c+1.
        
        def get_block_indices(r, c):
            return [2*r*4 + 2*c, 2*r*4 + 2*c + 1, (2*r+1)*4 + 2*c, (2*r+1)*4 + 2*c + 1]
        
        # Meta multiplications definition:
        # (A_blocks, coeffs_A, B_blocks, coeffs_B, C_block, coeff_C)
        # We will iterate through the 7 meta-multiplications.
        
        meta_mults = [
            # M1
            ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], (0,0), 1),
            # M2
            ([(1,0), (1,1)], [1, 1], [(0,0)], [1], (1,0), 1),
            # M3
            ([(0,0)], [1], [(0,1), (1,1)], [1, -1], (0,1), 1),
            # M4
            ([(1,1)], [1], [(1,0), (0,0)], [1, -1], (0,0), -1),
            # M5
            ([(0,0), (0,1)], [1, 1], [(1,1)], [1], (0,0), -1),
            # M6
            ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], (1,1), 1),
            # M7
            ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], (0,0), 1),
        ]
        
        # Also M1 contributes to C11
        # M4 contributes to C10
        # M5 contributes to C01
        # M7 contributes to C00 (already handled)
        
        # Wait, the W matrix of the 2x2x2 decomposition handles the combination.
        # The "meta" W matrix just places the 2x2 result block.
        # But the 2x2x2 decomposition's W matrix has coefficients for C00, C01, C10, C11.
        # So we need to be careful.
        
        # Let's re-evaluate.
        # The 2x2x2 decomposition computes C = A*B.
        # In the meta-step, we compute M = A_meta * B_meta.
        # A_meta is a 2x2 matrix (linear combo of A blocks).
        # B_meta is a 2x2 matrix (linear combo of B blocks).
        # M is a 2x2 matrix.
        # Then C_block = coeff * M.
        
        # So for each meta-multiplication, we instantiate the 2x2x2 decomposition.
        # The U matrix of the 2x2x2 decomposition operates on the 4 elements of A_meta.
        # But A_meta is formed from A blocks.
        # So U_4x4x4[row, col] = sum of U_2x2x2[local_row, col] * coeff_A[local_row]
        
        # Let's do this.
        
        col_offset = 0
        for a_blocks, a_coeffs, b_blocks, b_coeffs, c_block, c_coeff in meta_mults:
            # We need to map the 4 local indices of A_meta to the global indices of A.
            # A_meta is 2x2. Local indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1) of A_meta.
            # But A_meta is formed by summing blocks.
            # This is tricky because A_meta(0,0) = sum(coeff * A_block(r,c)(0,0)).
            
            # Actually, it's easier to think of it as:
            # The 2x2x2 decomposition has U, V, W.
            # We substitute A_local = sum a_coeffs[k] * A_block_k.
            # B_local = sum b_coeffs[k] * B_block_k.
            # C_local = sum W_2x2 * M_local.
            # And C_global_block = c_coeff * C_local.
            
            # So U_global[global_idx, col] = sum_k a_coeffs[k] * U_2x2[local_idx, col]
            # where global_idx corresponds to the element in A_block_k that maps to local_idx.
            
            # Let's build the mapping.
            # For each term in the sum for A_meta:
            # A_meta[i,j] = sum_k a_coeffs[k] * A_block_k[i,j]
            # So the coefficient of A_block_k[i,j] in A_meta[i,j] is a_coeffs[k].
            
            # U_2x2 has shape 4x7. Rows are 0,1,2,3 for A_meta elements.
            # For each row r in U_2x2 (0..3), it corresponds to A_meta[r//2, r%2].
            # We need to add U_2x2[r, c] * a_coeffs[k] to U_global[global_idx, col_offset+c]
            # where global_idx is the index of A_block_k[r//2, r%2].
            
            for r in range(4):
                meta_r = r // 2
                meta_c = r % 2
                for k, coeff in enumerate(a_coeffs):
                    br, bc = a_blocks[k]
                    global_idx = get_block_indices(br, bc)[r] # Wait, r is 0..3, block is 2x2.
                    # Block indices are [br*4+bc*2, ...]
                    # Actually, get_block_indices returns 4 indices.
                    # r is the index in the 2x2 block.
                    # So global_idx = get_block_indices(br, bc)[r]
                    U[global_idx, col_offset + r] += U_2x2[r, r] * coeff # Wrong col index
                    # U_2x2 is 4x7. We are iterating over the 7 columns of U_2x2.
            
            # Let's restart the loop structure.
            for c_local in range(7):
                col = col_offset + c_local
                # U contribution
                for r_local in range(4):
                    val = U_2x2[r_local, c_local]
                    if val != 0:
                        meta_r = r_local // 2
                        meta_c = r_local % 2
                        for k, coeff in enumerate(a_coeffs):
                            br, bc = a_blocks[k]
                            global_idx = get_block_indices(br, bc)[r_local]
                            U[global_idx, col] += val * coeff
                
                # V contribution
                for r_local in range(4):
                    val = V_2x2[r_local, c_local]
                    if val != 0:
                        meta_r = r_local // 2
                        meta_c = r_local % 2
                        for k, coeff in enumerate(b_coeffs):
                            br, bc = b_blocks[k]
                            global_idx = get_block_indices(br, bc)[r_local]
                            V[global_idx, col] += val * coeff
                
                # W contribution
                # W_2x2 maps M_local to C_local.
                # C_global_block = c_coeff * C_local.
                # So W_global[global_idx, col] += c_coeff * W_2x2[r_local, c_local]
                # where global_idx is in C_block.
                for r_local in range(4):
                    val = W_2x2[r_local, c_local]
                    if val != 0:
                        cr, cc = c_block
                        global_idx = get_block_indices(cr, cc)[r_local]
                        W[global_idx, col] += val * c_coeff
            
            col_offset += 7
        
        return U, V, W

    def strassen_4x4x3():
        # Returns U, V, W for 4x4x3
        # U: 16x44
        # V: 12x44 (4x3 flat)
        # W: 12x44 (4x3 flat)
        
        U = np.zeros((16, 44))
        V = np.zeros((12, 44))
        W = np.zeros((12, 44))
        
        # Split into 4 blocks of 2x2x3.
        # A is 4x4. Blocks A00, A01, A10, A11.
        # B is 4x3. Blocks B_top (2x3), B_bot (2x3).
        # C is 4x3. Blocks C_top (2x3), C_bot (2x3).
        
        # C_top = A00 * B_top + A01 * B_bot
        # C_bot = A10 * B_top + A11 * B_bot
        
        # We need to compute 2x2x3 products.
        # 2x2x3 rank is 11.
        # We have 4 such products.
        # Total rank 44.
        
        # We need a 2x2x3 decomposition.
        # It can be built from 2x2x2 (rank 7) and 2x2x1 (rank 4).
        # B (2x3) = [B_2x2, B_2x1].
        # C (2x3) = [C_2x2, C_2x1].
        # C_2x2 = A * B_2x2 (rank 7)
        # C_2x1 = A * B_2x1 (rank 4)
        
        def strassen_2x2x3():
            U_22 = np.zeros((4, 11))
            V_22 = np.zeros((6, 11)) # 2x3 flat
            W_22 = np.zeros((6, 11))
            
            U_2, V_2, W_2 = strassen_2x2x2()
            
            # 2x2x2 part (cols 0-6)
            U_22[:, 0:7] = U_2
            V_22[0:4, 0:7] = V_2 # B_2x2 is first 4 elements
            W_22[0:4, 0:7] = W_2 # C_2x2 is first 4 elements
            
            # 2x2x1 part (cols 7-10)
            # A (2x2) * B (2x1) = C (2x1)
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            # Rank 4.
            U_22[0, 7] = 1
            V_22[4, 7] = 1
            W_22[4, 7] = 1 # C0 (index 4 in 2x3? No, C_2x1 is last 2 elements)
            # Flat indices for 2x3:
            # 0: (0,0), 1: (0,1), 2: (0,2)
            # 3: (1,0), 4: (1,1), 5: (1,2)
            # B_2x2 is (0,0),(0,1),(1,0),(1,1) -> indices 0,1,3,4?
            # Wait, standard flat order is row-major.
            # 2x3:
            # 0 1 2
            # 3 4 5
            # B_2x2 is cols 0,1. Indices 0,1,3,4.
            # B_2x1 is col 2. Indices 2,5.
            
            # So V_2x2 (2x2 part) uses indices 0,1,3,4.
            # V_2x1 (2x1 part) uses indices 2,5.
            
            # Correct V_22 mapping:
            V_22[0, 0:7] = V_2[0, 0:7] # B00
            V_22[1, 0:7] = V_2[1, 0:7] # B01
            V_22[3, 0:7] = V_2[2, 0:7] # B10
            V_22[4, 0:7] = V_2[3, 0:7] # B11
            
            W_22[0, 0:7] = W_2[0, 0:7] # C00
            W_22[1, 0:7] = W_2[1, 0:7] # C01
            W_22[3, 0:7] = W_2[2, 0:7] # C10
            W_22[4, 0:7] = W_2[3, 0:7] # C11
            
            # 2x2x1 part
            U_22[0, 7] = 1
            V_22[2, 7] = 1 # B02
            W_22[2, 7] = 1 # C02
            
            U_22[1, 8] = 1
            V_22[5, 8] = 1 # B12
            W_22[2, 8] = 1 # C02
            
            U_22[2, 9] = 1
            V_22[2, 9] = 1
            W_22[5, 9] = 1 # C12
            
            U_22[3, 10] = 1
            V_22[5, 10] = 1
            W_22[5, 10] = 1
            
            return U_22, V_22, W_22
        
        U_23, V_23, W_23 = strassen_2x2x3()
        
        # Now assemble 4x4x3.
        # Products:
        # 1. A00 * B_top
        # 2. A01 * B_bot
        # 3. A10 * B_top
        # 4. A11 * B_bot
        
        # B_top is rows 0,1 of B (4x3). Indices 0,1,2, 3,4,5.
        # B_bot is rows 2,3 of B. Indices 6,7,8, 9,10,11.
        
        # C_top is rows 0,1 of C. Indices 0,1,2, 3,4,5.
        # C_bot is rows 2,3 of C. Indices 6,7,8, 9,10,11.
        
        def get_a_block_indices(r, c):
            # A is 4x4. Block (r,c) is 2x2.
            # Rows 2r, 2r+1. Cols 2c, 2c+1.
            return [2*r*4 + 2*c, 2*r*4 + 2*c + 1, (2*r+1)*4 + 2*c, (2*r+1)*4 + 2*c + 1]
        
        def get_b_block_indices(row_offset):
            # B is 4x3. Block is 2x3.
            # Rows row_offset, row_offset+1. Cols 0,1,2.
            # Flat indices:
            # (row_offset, 0), (row_offset, 1), (row_offset, 2)
            # (row_offset+1, 0), (row_offset+1, 1), (row_offset+1, 2)
            return [row_offset*3 + 0, row_offset*3 + 1, row_offset*3 + 2, 
                    (row_offset+1)*3 + 0, (row_offset+1)*3 + 1, (row_offset+1)*3 + 2]
        
        def get_c_block_indices(row_offset):
            # Same as B
            return [row_offset*3 + 0, row_offset*3 + 1, row_offset*3 + 2, 
                    (row_offset+1)*3 + 0, (row_offset+1)*3 + 1, (row_offset+1)*3 + 2]
        
        products = [
            (0, 0, 0, 0), # A00, B_top, C_top
            (1, 1, 0, 1), # A01, B_bot, C_top
            (2, 0, 2, 0), # A10, B_top, C_bot
            (3, 1, 3, 1), # A11, B_bot, C_bot
        ]
        
        col_offset = 0
        for a_idx, b_row_off, c_idx, c_row_off in products:
            # a_idx is 0..3 for A00, A01, A10, A11
            # Map a_idx to (r,c)
            ar = a_idx // 2
            ac = a_idx % 2
            
            a_indices = get_a_block_indices(ar, ac)
            b_indices = get_b_block_indices(b_row_off)
            c_indices = get_c_block_indices(c_row_off)
            
            for c_local in range(11):
                col = col_offset + c_local
                for r_local in range(4):
                    val = U_23[r_local, c_local]
                    if val != 0:
                        U[a_indices[r_local], col] = val
                for r_local in range(6):
                    val = V_23[r_local, c_local]
                    if val != 0:
                        V[b_indices[r_local], col] = val
                for r_local in range(6):
                    val = W_23[r_local, c_local]
                    if val != 0:
                        W[c_indices[r_local], col] = val
            col_offset += 11
            
        return U, V, W

    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = strassen_4x4x3()
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.vstack([V1, V2]) # V1 is 16x49, V2 is 12x44. Wait.
    
    # V1 corresponds to B1 (4x4). V2 corresponds to B2 (4x3).
    # B is 4x7. B = [B1, B2].
    # So V should be (4*7) x Rank.
    # V1 is 16 x 49.
    # V2 is 12 x 44.
    # We need to embed V1 and V2 into a 28 x 93 matrix.
    
    V_full = np.zeros((28, 93))
    V_full[0:16, 0:49] = V1
    V_full[16:28, 49:93] = V2
    
    W_full = np.zeros((28, 93))
    W_full[0:16, 0:49] = W1
    W_full[16:28, 49:93] = W2
    
    return {"U": U.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
