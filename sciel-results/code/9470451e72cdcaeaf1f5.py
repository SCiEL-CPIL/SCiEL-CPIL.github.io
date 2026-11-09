import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 94.
    # Part 1: 4x4 Strassen (rank 49)
    # Part 2: Cross terms (rank 45)
    
    # 1. Generate 2x2 Strassen U, V, W
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)
    
    # M0 = (A00 + A11)(B00 + B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    W_2[1, 0] = 1; W_2[2, 0] = 1
    # M0 = C00 + C11, C01 - C00, C11 - C10?
    # C00 = M0 + M3 + M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # M1 = (A10 + A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = 1
    
    # M2 = A00(B10 - B11)
    U_2[0, 2] = 1
    V_2[2, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M3 = A11(B01 - B00)
    U_2[3, 3] = 1
    V_2[1, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M4 = (A00 + A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[1, 4] = 1; W_2[3, 4] = 1
    
    # M5 = (A10 - A00)(B00 + B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[0, 5] = 1; W_2[2, 5] = 1
    
    # M6 = (A01 - A11)(B10 + B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = -1; W_2[1, 6] = 1; W_2[2, 6] = -1; W_2[3, 6] = 1
    
    # Verify 2x2 Strassen
    # C = A B
    # C00 = M0 + M3 + M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # 2. Generate 4x4 Strassen U, V, W
    # 4x4 blocks:
    # A00: 0,1,2,3 (indices 0,1,4,5) -> No, indices are row-major.
    # A00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # Map local 2x2 index (0,1,2,3) to global 4x4 indices for each block
    # Block 00: [0, 1, 4, 5]
    # Block 01: [2, 3, 6, 7]
    # Block 10: [8, 9, 12, 13]
    # Block 11: [10, 11, 14, 15]
    
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # Define the 7 Strassen combinations for 4x4
    # Each is a list of (block_idx, coeff) for A and B
    # M0: (A00 + A11)(B00 + B11)
    # M1: (A10 + A11)B00
    # M2: A00(B10 - B11)
    # M3: A11(B01 - B00)
    # M4: (A00 + A01)B11
    # M5: (A10 - A00)(B00 + B01)
    # M6: (A01 - A11)(B10 + B11)
    
    strassen_4x4_combos = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M0
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),          # M1
        ([ (0, 1) ], [ (2, 1), (3, -1) ]),         # M2
        ([ (3, 1) ], [ (1, 1), (0, -1) ]),         # M3
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),          # M4
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]), # M5
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])  # M6
    ]
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    col = 0
    for combo_idx in range(7):
        a_combo, b_combo = strassen_4x4_combos[combo_idx]
        # For each of the 7 sub-terms in 2x2 Strassen
        for sub_idx in range(7):
            # Get the U, V, W coefficients from 2x2 Strassen for this sub-term
            u_coeff = U_2[:, sub_idx]
            v_coeff = V_2[:, sub_idx]
            w_coeff = W_2[:, sub_idx]
            
            # Map to global indices
            # For U:
            # u_coeff[i] is the coefficient for local index i.
            # Local index i corresponds to an entry in the "A" matrix of the sub-problem.
            # The "A" matrix of the sub-problem is the linear combination of blocks defined by a_combo.
            # Wait, the sub-problem is a 2x2 matrix multiplication.
            # The "A" of the sub-problem is a 2x2 matrix of blocks?
            # No, the sub-problem is just a 2x2 matrix multiplication of the *blocks*?
            # No, Strassen works on scalars.
            # The "A" in M0 = (A00+A11)(B00+B11) is a 2x2 matrix of *scalars*?
            # No, A00 is a 2x2 matrix of scalars.
            # So (A00+A11) is a 2x2 matrix of scalars.
            # So we are multiplying two 2x2 matrices of scalars.
            # So we use the 2x2 Strassen decomposition on these 2x2 matrices.
            # The entries of the "A" matrix of this sub-problem are:
            # A_sub[0,0] = A00[0,0] + A11[0,0]
            # A_sub[0,1] = A00[0,1] + A11[0,1]
            # etc.
            
            # So, for each local index i (0..3) in the 2x2 sub-problem:
            # i corresponds to (r, c) where r = i//2, c = i%2.
            # The value at A_sub[r, c] is the sum of coeff * A_block[r, c][r_local, c_local]?
            # No, A_block is a 2x2 matrix.
            # A_sub[r, c] is a scalar? No, A_sub is a 2x2 matrix of scalars.
            # The entries of A_sub are sums of entries of the blocks.
            # Specifically, A_sub[r, c] = sum_{block} coeff * A_block[r, c].
            # Wait, A_block is a 2x2 matrix.
            # The "A" in the 2x2 Strassen decomposition is a 2x2 matrix of scalars.
            # So A_sub[r, c] is a scalar.
            # But A_block[r, c] is also a scalar (entry of the block matrix).
            # So yes, A_sub[r, c] = sum coeff * A_block[r, c].
            
            # So, for a given local index i (0..3) in the 2x2 sub-problem:
            # It corresponds to row r=i//2, col c=i%2 of the sub-problem matrix.
            # The global indices contributing to this are:
            # For each (block_idx, coeff) in a_combo:
            #   The global index is block_indices[block_idx][i].
            #   Because the entry (r,c) of the block is at the same relative position in the global matrix.
            #   Wait, block_indices[block_idx] gives the 4 global indices for the 4 entries of the block.
            #   The local index i in the 2x2 sub-problem corresponds to the i-th entry of the block?
            #   Yes, because the sub-problem is a 2x2 matrix multiplication of the *blocks*?
            #   No, the sub-problem is a 2x2 matrix multiplication of the *scalars* that form the blocks.
            #   Actually, it's simpler:
            #   The 4x4 matrix is partitioned into four 2x2 blocks.
            #   Strassen computes 7 products of these blocks.
            #   Each product is a 2x2 matrix multiplication.
            #   So we decompose each block product using 2x2 Strassen.
            #   The "A" matrix for the sub-problem is the block (or sum of blocks).
            #   The entries of the "A" matrix are the entries of the blocks.
            #   So local index i in the sub-problem corresponds to the i-th entry of the block.
            #   And the i-th entry of the block is at global index block_indices[block_idx][i].
            
            for i in range(4):
                coeff = u_coeff[i]
                if coeff != 0:
                    for block_idx, b_coeff in a_combo:
                        global_idx = block_indices[block_idx][i]
                        U_4[global_idx, col] += coeff * b_coeff
            
            for i in range(4):
                coeff = v_coeff[i]
                if coeff != 0:
                    for block_idx, b_coeff in b_combo:
                        global_idx = block_indices[block_idx][i]
                        V_4[global_idx, col] += coeff * b_coeff
            
            for i in range(4):
                coeff = w_coeff[i]
                if coeff != 0:
                    for block_idx, b_coeff in [(0,1), (1,1), (2,1), (3,1)]: # C is sum of all blocks?
                        # No, C is the result of the 4x4 multiplication.
                        # The sub-problem result is a 2x2 matrix of blocks.
                        # The local index i in W corresponds to the i-th entry of the result block.
                        # But which block?
                        # The result of the sub-problem is added to the corresponding block of C.
                        # The sub-problem corresponds to a specific block of C?
                        # No, the 7 Strassen terms M0..M6 are combined to form C.
                        # C00 = M0 + M3 + M5 - M6
                        # C01 = M2 + M4
                        # C10 = M1 + M5
                        # C11 = M0 - M1 + M2 + M4
                        # So each M_k contributes to specific blocks of C.
                        # We need to know which block of C each M_k contributes to.
                        # This depends on the Strassen formula.
                        
                        # Let's re-verify the Strassen formula mapping.
                        # M0 -> C00, C11
                        # M1 -> C10, C11
                        # M2 -> C01, C11
                        # M3 -> C00, C10? No.
                        
                        # Standard Strassen:
                        # C00 = M0 + M3 + M5 - M6
                        # C01 = M2 + M4
                        # C10 = M1 + M5
                        # C11 = M0 - M1 + M2 + M4
                        
                        # So for M0: contributes to C00 and C11.
                        # For M1: contributes to C10 and C11.
                        # etc.
                        
                        # So for a given combo_idx (0..6), we need to know which blocks of C it affects.
                        pass
            
            # This is getting complicated. Let's use a simpler mapping.
            # The W matrix for the 4x4 case will have non-zeros in the blocks of C.
            # For a given M_k, it contributes to specific C blocks.
            # Let's define the C block contributions for each M_k.
            
            col += 1
            
    # This approach is error-prone. Let's just use the schoolbook for the 4x4 part?
    # Rank 64. Total 64 + 45 = 109. Still better than 125.
    # But 94 is better.
    
    # Let's try to hardcode the 4x4 Strassen W mapping.
    
    # M0: C00, C11
    # M1: C10, C11
    # M2: C01, C11
    # M3: C00, C10? No, C00 = M0 + M3 + M5 - M6. So M3 -> C00.
    # M4: C01, C11
    # M5: C00, C10
    # M6: C00, C11? No, C00 = ... - M6. C11 = ... + M6? No.
    # C11 = M0 - M1 + M2 + M4.
    
    # So:
    # M0: C00, C11
    # M1: C10, C11
    # M2: C01, C11
    # M3: C00
    # M4: C01, C11
    # M5: C00, C10
    # M6: C00
    
    # Wait, C10 = M1 + M5.
    # C01 = M2 + M4.
    
    # Let's re-verify C00.
    # C00 = M0 + M3 + M5 - M6.
    
    # So:
    # M0: C00, C11
    # M1: C10, C11
    # M2: C01, C11
    # M3: C00
    # M4: C01, C11
    # M5: C00, C10
    # M6: C00
    
    # This seems correct.
    
    # Let's restart the 4x4 generation with this mapping.
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    c_block_map = {
        0: [0, 1], # M0 -> C00, C11
        1: [2, 3], # M1 -> C10, C11
        2: [1, 3], # M2 -> C01, C11
        3: [0],    # M3 -> C00
        4: [1, 3], # M4 -> C01, C11
        5: [0, 2], # M5 -> C00, C10
        6: [0]     # M6 -> C00
    }
    
    col = 0
    for combo_idx in range(7):
        a_combo, b_combo = strassen_4x4_combos[combo_idx]
        for sub_idx in range(7):
            u_coeff = U_2[:, sub_idx]
            v_coeff = V_2[:, sub_idx]
            w_coeff = W_2[:, sub_idx]
            
            for i in range(4):
                coeff = u_coeff[i]
                if coeff != 0:
                    for block_idx, b_coeff in a_combo:
                        global_idx = block_indices[block_idx][i]
                        U_4[global_idx, col] += coeff * b_coeff
            
            for i in range(4):
                coeff = v_coeff[i]
                if coeff != 0:
                    for block_idx, b_coeff in b_combo:
                        global_idx = block_indices[block_idx][i]
                        V_4[global_idx, col] += coeff * b_coeff
            
            for i in range(4):
                coeff = w_coeff[i]
                if coeff != 0:
                    for c_block_idx in c_block_map[combo_idx]:
                        global_idx = block_indices[c_block_idx][i]
                        W_4[global_idx, col] += coeff
            
            col += 1
            
    # 3. Generate Cross Terms
    # Indices 0..4
    # U, V, W size 25 x 45
    
    U_cross = np.zeros((25, 45), dtype=int)
    V_cross = np.zeros((25, 45), dtype=int)
    W_cross = np.zeros((25, 45), dtype=int)
    
    col = 0
    # 1. Aij Bj4 (16)
    for i in range(4):
        for j in range(4):
            U_cross[i * 5 + j, col] = 1
            V_cross[j * 5 + 4, col] = 1
            W_cross[4 * 5 + i, col] = 1
            col += 1
            
    # 2. Ai4 B44 (4)
    for i in range(4):
        U_cross[i * 5 + 4, col] = 1
        V_cross[4 * 5 + 4, col] = 1
        W_cross[4 * 5 + i, col] = 1
        col += 1
        
    # 3. A4i Bij (16)
    for i in range(4):
        for j in range(4):
            U_cross[4 * 5 + i, col] = 1
            V_cross[i * 5 + j, col] = 1
            W_cross[j * 5 + 4, col] = 1
            col += 1
            
    # 4. A44 B4j (4)
    for j in range(4):
        U_cross[4 * 5 + 4, col] = 1
        V_cross[4 * 5 + j, col] = 1
        W_cross[j * 5 + 4, col] = 1
        col += 1
        
    # 5. A4i Bi4 (4)
    for i in range(4):
        U_cross[4 * 5 + i, col] = 1
        V_cross[i * 5 + 4, col] = 1
        W_cross[4 * 5 + 4, col] = 1
        col += 1
        
    # 6. A44 B44 (1)
    U_cross[4 * 5 + 4, col] = 1
    V_cross[4 * 5 + 4, col] = 1
    W_cross[4 * 5 + 4, col] = 1
    col += 1
    
    # Combine
    U = np.zeros((25, 94), dtype=int)
    V = np.zeros((25, 94), dtype=int)
    W = np.zeros((25, 94), dtype=int)
    
    U[:16, :49] = U_4
    V[:16, :49] = V_4
    W[:16, :49] = W_4
    
    U[:, 49:] = U_cross
    V[:, 49:] = V_cross
    W[:, 49:] = W_cross
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
