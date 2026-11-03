import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build a rank-53 decomposition.
    # Part 1: Rank-49 for 4x4x4
    # Part 2: Rank-4 for 4x4x1 (last column)
    
    rank_4x4_4x4 = 49
    rank_4x4_1 = 4
    total_rank = rank_4x4_4x4 + rank_4x4_1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 ---
    # Generate 49 terms
    
    # Helper: 2x2 Strassen terms
    # U, V, W are length 4 (indices 0..3 for 2x2)
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00, C11
    # Term 1: M2 = (A10+A11)B00 -> C10
    # Term 2: M3 = A00(B01-B11) -> C01
    # Term 3: M4 = A11(B10-B00) -> C10, C00
    # Term 4: M5 = (A00+A01)B11 -> C01, C00
    # Term 5: M6 = (A10-A00)(B00+B01) -> C11
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00, C11
    
    # Let's define the 7 terms for 2x2x2
    # Each term is (U_vec, V_vec, W_vec)
    s2x2_terms = []
    
    # M1
    u = [0]*4; v = [0]*4; w = [0]*4
    u[0] = 1; u[3] = 1
    v[0] = 1; v[3] = 1
    w[0] = 1; w[3] = 1 # C00, C11
    s2x2_terms.append((u, v, w))
    
    # M2
    u = [0]*4; v = [0]*4; w = [0]*4
    u[2] = 1; u[3] = 1 # A10, A11
    v[0] = 1           # B00
    w[2] = 1           # C10
    s2x2_terms.append((u, v, w))
    
    # M3
    u = [0]*4; v = [0]*4; w = [0]*4
    u[0] = 1           # A00
    v[1] = 1; v[3] = -1 # B01, B11
    w[1] = 1           # C01
    s2x2_terms.append((u, v, w))
    
    # M4
    u = [0]*4; v = [0]*4; w = [0]*4
    u[3] = 1           # A11
    v[2] = 1; v[0] = -1 # B10, B00
    w[2] = 1; w[0] = 1  # C10, C00
    s2x2_terms.append((u, v, w))
    
    # M5
    u = [0]*4; v = [0]*4; w = [0]*4
    u[0] = 1; u[1] = 1 # A00, A01
    v[3] = 1           # B11
    w[1] = 1; w[0] = -1 # C01, C00
    s2x2_terms.append((u, v, w))
    
    # M6
    u = [0]*4; v = [0]*4; w = [0]*4
    u[2] = 1; u[0] = -1 # A10, A00
    v[0] = 1; v[1] = 1  # B00, B01
    w[3] = 1            # C11
    s2x2_terms.append((u, v, w))
    
    # M7
    u = [0]*4; v = [0]*4; w = [0]*4
    u[1] = 1; u[3] = -1 # A01, A11
    v[2] = 1; v[3] = 1  # B10, B11
    w[0] = 1; w[3] = -1 # C00, C11
    s2x2_terms.append((u, v, w))
    
    # Now construct 4x4x4 terms
    # We have 7 "major" Strassen steps for the 4x4 matrix.
    # Each major step computes a product of two 2x2 block matrices.
    # Let's define the 7 major combinations.
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    major_combos = []
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00, A11. B part: B00, B11.
    # Coeffs for A blocks: [1, 0, 0, 1]
    # Coeffs for B blocks: [1, 0, 0, 1]
    # Output contributes to C00, C11 with coeff 1.
    major_combos.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (A10+A11)B00
    # A part: A10, A11. B part: B00.
    # Output contributes to C10 with coeff 1.
    major_combos.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # M3 = A00(B01-B11)
    # A part: A00. B part: B01, B11.
    # Output contributes to C01 with coeff 1.
    major_combos.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    
    # M4 = A11(B10-B00)
    # A part: A11. B part: B10, B00.
    # Output contributes to C10, C00 with coeff 1.
    major_combos.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    
    # M5 = (A00+A01)B11
    # A part: A00, A01. B part: B11.
    # Output contributes to C01, C00 with coeff -1 for C00?
    # Formula: C00 = M1 + M4 - M5 + M7. So -1 for C00.
    # Formula: C01 = M3 + M5. So +1 for C01.
    major_combos.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6 = (A10-A00)(B00+B01)
    # A part: A10, A00. B part: B00, B01.
    # Output contributes to C11 with coeff 1.
    major_combos.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    
    # M7 = (A01-A11)(B10+B11)
    # A part: A01, A11. B part: B10, B11.
    # Output contributes to C00, C11 with coeff -1 for C11?
    # Formula: C00 = ... + M7. So +1 for C00.
    # Formula: C11 = ... + M6. No M7 in C11?
    # Check Strassen again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M7 only in C00.
    major_combos.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # Now iterate through major combos and sub-terms
    col_idx = 0
    for mc in major_combos:
        a_coeffs, b_coeffs, c_coeffs = mc
        # a_coeffs: [A00, A01, A10, A11]
        # b_coeffs: [B00, B01, B10, B11]
        # c_coeffs: [C00, C01, C10, C11]
        
        for sub_u, sub_v, sub_w in s2x2_terms:
            # sub_u, sub_v, sub_w are length 4 vectors for the 2x2 virtual multiplication.
            # sub_u[i] is coeff for virtual A_block[i]
            # We need to expand this to 4x4 indices.
            
            u_4x4 = [0] * 16
            v_4x4 = [0] * 16
            w_4x4 = [0] * 16
            
            # Expand U
            for block_idx in range(4):
                if a_coeffs[block_idx] != 0:
                    # block_idx maps to 2x2 position
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    br = block_idx // 2
                    bc = block_idx % 2
                    # The virtual input to this block is multiplied by a_coeffs[block_idx]
                    # And the sub-term has coefficients sub_u[?]
                    # Wait, the sub-term coefficients sub_u are for the elements of the virtual 2x2 matrix.
                    # The virtual 2x2 matrix elements ARE the blocks.
                    # So virtual element (r,c) corresponds to block (br, bc) where r=br, c=bc?
                    # Yes.
                    # So coeff for block (br, bc) is a_coeffs[br*2+bc] * sub_u[br*2+bc]?
                    # No. sub_u is the coefficient for the *entire* block in the rank-1 decomposition of the block multiplication.
                    # Actually, the rank-1 decomposition of the block multiplication treats the blocks as scalars.
                    # So sub_u[br*2+bc] is the weight for the block at (br, bc).
                    # The block at (br, bc) is a 2x2 matrix of original elements.
                    # The element (i,j) in the original 4x4 matrix corresponding to block (br, bc) and local (lr, lc) is:
                    # i = br*2 + lr
                    # j = bc*2 + lc
                    # The rank-1 term for the block multiplication assigns weight sub_u[br*2+bc] to the *whole block*?
                    # No, a rank-1 term in the block space is $u_{block} v_{block} w_{block}$.
                    # This expands to $\sum_{lr, lc} u_{block} A_{br, bc, lr, lc} \dots$
                    # So the weight for each element in the block is $a\_coeffs[br*2+bc] \times sub\_u[br*2+bc]$.
                    
                    weight = a_coeffs[block_idx] * sub_u[block_idx]
                    if weight != 0:
                        for lr in range(2):
                            for lc in range(2):
                                i = br * 2 + lr
                                j = bc * 2 + lc
                                u_4x4[i * 4 + j] += weight
                                
            # Expand V
            for block_idx in range(4):
                if b_coeffs[block_idx] != 0:
                    br = block_idx // 2
                    bc = block_idx % 2
                    weight = b_coeffs[block_idx] * sub_v[block_idx]
                    if weight != 0:
                        for lr in range(2):
                            for lc in range(2):
                                j = br * 2 + lr
                                k = bc * 2 + lc
                                v_4x4[j * 4 + k] += weight
                                
            # Expand W
            for block_idx in range(4):
                if c_coeffs[block_idx] != 0:
                    br = block_idx // 2
                    bc = block_idx % 2
                    weight = c_coeffs[block_idx] * sub_w[block_idx]
                    if weight != 0:
                        for lr in range(2):
                            for lc in range(2):
                                k = br * 2 + lr
                                i = bc * 2 + lc
                                w_4x4[k * 4 + i] += weight
                                
            # Add to global U, V, W
            for r in range(16):
                U[r][col_idx] = u_4x4[r]
                V[r][col_idx] = v_4x4[r]
                W[r][col_idx] = w_4x4[r]
            
            col_idx += 1
            
    # --- Part 2: 4x4x1 ---
    # Last column of B (index k=4)
    # C_i = sum_j A_ij B_j4
    for j in range(4):
        for i in range(4):
            U[i * 4 + j][col_idx] = 1
            V[j * 5 + 4][col_idx] = 1
            W[4 * 4 + i][col_idx] = 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
