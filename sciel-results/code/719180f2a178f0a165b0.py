"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=8, m=4. Split B into two 4x4 matrices.
    # We use a recursive Strassen-based decomposition of rank 49 for 4x4x4.
    # Total rank = 49 + 49 = 98.
    
    # 1. Define Strassen 2x2 factors (Rank 7)
    # U2, V2, W2 are 4x7 matrices.
    # Rows 0-3 correspond to A/B elements in row-major order:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Verify W2 against C = AB
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check M1 (col 0): A=(1,0,0,1), B=(1,0,0,1). 
    # Wait, my U2/V2 definitions must match the Strassen products I listed earlier.
    # Let's re-verify U2/V2/W2 consistency.
    # My U2 above:
    # Col 0: A00=1, A11=1. B00=1, B11=1. M1.
    # Col 1: A10=1, A11=1. B00=1. M2.
    # Col 2: A00=1. B01=1, B11=-1. M3.
    # Col 3: A11=1. B10=1, B00=-1. M4.
    # Col 4: A00=1, A01=1. B11=1. M5.
    # Col 5: A10=1, A00=-1. B00=1, B01=1. M6.
    # Col 6: A01=1, A11=-1. B10=1, B11=1. M7.
    
    # My W2 above:
    # Row 0 (C00): 1, 0, 0, 1, -1, 0, 1. Matches C00 = M1 + M4 - M5 + M7.
    # Row 1 (C01): 0, 0, 1, 0, 1, 0, 0. Matches C01 = M3 + M5.
    # Row 2 (C10): 0, 1, 0, 1, 0, 0, 0. Matches C10 = M2 + M4.
    # Row 3 (C11): 1, -1, 1, 0, 0, 1, 0. Matches C11 = M1 - M2 + M3 + M6.
    
    # Consistency check passed.
    
    # 2. Construct 4x4x4 factors (Rank 49)
    # Blocks:
    # A00: idx 0,1,4,5. Local: 0,1,2,3?
    # Row-major:
    # 0: 0,0. 1: 0,1. 2: 0,2. 3: 0,3.
    # 4: 1,0. 5: 1,1. 6: 1,2. 7: 1,3.
    # 8: 2,0. 9: 2,1. 10: 2,2. 11: 2,3.
    # 12: 3,0. 13: 3,1. 14: 3,2. 15: 3,3.
    
    # Block mapping for A (and B):
    # Block 00 (Rows 0-1, Cols 0-1): Indices 0,1,4,5. Local 0,1,2,3.
    # Block 01 (Rows 0-1, Cols 2-3): Indices 2,3,6,7. Local 0,1,2,3.
    # Block 10 (Rows 2-3, Cols 0-1): Indices 8,9,12,13. Local 0,1,2,3.
    # Block 11 (Rows 2-3, Cols 2-3): Indices 10,11,14,15. Local 0,1,2,3.
    
    # Let's create a mapping array: global_idx -> (block_id, local_idx)
    # block_id: 0=00, 1=01, 2=10, 3=11.
    # local_idx: 0=TL, 1=TR, 2=BL, 3=BR.
    
    def get_block_info(idx):
        r = idx // 4
        c = idx % 4
        br = 0 if r < 2 else 1
        bc = 0 if c < 2 else 1
        block_id = br * 2 + bc
        lr = r % 2
        lc = c % 2
        local_idx = lr * 2 + lc
        return block_id, local_idx

    block_info = [get_block_info(i) for i in range(16)]

    # Strassen Outer Coefficients for 4x4
    # L_j = sum alpha_b A_b
    # R_j = sum beta_b B_b
    
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # Coefficients alpha[b] for L_j, beta[b] for R_j.
    # b: 0=00, 1=01, 2=10, 3=11.
    
    outer_alphas = [
        [1, 0, 0, 1], # P1: A00 + A11
        [0, 0, 1, 1], # P2: A10 + A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00 + A01
        [-1, 0, 1, 0],# P6: A10 - A00
        [0, 1, 0, -1] # P7: A01 - A11
    ]
    
    outer_betas = [
        [1, 0, 0, 1], # P1: B00 + B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1],# P3: B01 - B11
        [-1, 0, 1, 0],# P4: B10 - B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00 + B01
        [0, 0, 1, 1]  # P7: B10 + B11
    ]
    
    # Gamma coefficients for C_blocks
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # gamma[block_id][j]
    outer_gammas = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]

    rank_49 = 49
    U4 = [[0] * rank_49 for _ in range(16)]
    V4 = [[0] * rank_49 for _ in range(16)]
    W4 = [[0] * rank_49 for _ in range(16)]
    
    for j in range(7):
        for k in range(7):
            col = j * 7 + k
            
            # Construct U4 column
            for g in range(16):
                b_id, l_id = block_info[g]
                U4[g][col] = outer_alphas[j][b_id] * U2[l_id][k]
                
            # Construct V4 column
            for g in range(16):
                b_id, l_id = block_info[g]
                V4[g][col] = outer_betas[j][b_id] * V2[l_id][k]
                
            # Construct W4 column
            for g in range(16):
                b_id, l_id = block_info[g]
                W4[g][col] = outer_gammas[b_id][j] * W2[l_id][k]
                
    # 3. Extend to 4x4x8
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C = A B_left + A B_right.
    # We duplicate the 49 columns, one set for B_left, one for B_right.
    # V matrix needs to handle 4x8 input (32 elements).
    # V shape: 32 x 98.
    # First 16 cols of V correspond to B_left? No, V rows correspond to input B.
    # B has 32 elements (4x8).
    # Rows 0-15: B_left (cols 0-3 of B).
    # Rows 16-31: B_right (cols 4-7 of B).
    
    # U shape: 16 x 98. (A is 4x4).
    # W shape: 32 x 98. (C is 4x8).
    # C rows 0-15: C_left (cols 0-3).
    # C rows 16-31: C_right (cols 4-7).
    
    total_rank = 98
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(32)]
    W = [[0] * total_rank for _ in range(32)]
    
    # Fill first 49 columns using B_left and C_left
    for c in range(49):
        # U is same for both halves (A is same)
        for i in range(16):
            U[i][c] = U4[i][c]
            U[i][c + 49] = U4[i][c]
            
        # V: Rows 0-15 are B_left, Rows 16-31 are B_right.
        # For the first half of columns, V acts on B_left.
        # For the second half, V acts on B_right.
        # Since the decomposition for the second half is identical in structure,
        # we just map the rows.
        
        for i in range(16):
            # B_left row i
            V[i][c] = V4[i][c]
            # B_right row 16+i
            V[16 + i][c + 49] = V4[i][c]
            
        # Zeros in V for the "other" half
        for i in range(16):
            V[16 + i][c] = 0
            V[i][c + 49] = 0
            
        # W: Rows 0-15 are C_left, Rows 16-31 are C_right.
        for i in range(16):
            W[i][c] = W4[i][c]
            W[16 + i][c + 49] = W4[i][c]
            
        for i in range(16):
            W[16 + i][c] = 0
            W[i][c + 49] = 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
