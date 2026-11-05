"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 4x4x4 (rank 49) and Schoolbook for 4x4x1 (rank 16)
    # Total rank = 65
    
    rank_strassen = 49
    rank_schoolbook = n * 4 * 1  # 4*4*1 = 16
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. Generate Strassen terms for 4x4x4
    # We need to map the 4x4 indices to 2x2 blocks, then to 1x1 elements
    
    # Strassen 2x2x2 terms (U, V, W vectors of length 4)
    # Indices: 0,1,2,3 correspond to [a,b,c,d]
    s2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # m3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # m7
    ]
    
    # For 4x4x4, we apply Strassen to 2x2 blocks.
    # Block indices:
    # A: 00, 01, 10, 11 (each is 2x2)
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Outer Strassen terms for 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the linear combinations for A and B blocks
    # Each is a dict {block_idx: coeff}
    outer_A = [
        {0: 1, 3: 1},   # M1: A00+A11
        {2: 1, 3: 1},   # M2: A10+A11
        {0: 1},         # M3: A00
        {3: 1},         # M4: A11
        {0: 1, 1: 1},   # M5: A00+A01
        {2: 1, 0: -1},  # M6: A10-A00
        {1: 1, 3: -1}   # M7: A01-A11
    ]
    outer_B = [
        {0: 1, 3: 1},   # M1: B00+B11
        {0: 1},         # M2: B00
        {1: 1, 3: -1},  # M3: B01-B11
        {2: 1, 0: -1},  # M4: B10-B00
        {3: 1},         # M5: B11
        {0: 1, 1: 1},   # M6: B00+B01
        {2: 1, 3: 1}    # M7: B10+B11
    ]
    
    # For each outer term k, and each inner Strassen term l:
    # The U vector is the sum of (coeff_A_block * s2_U_l_expanded)
    # The V vector is the sum of (coeff_B_block * s2_V_l_expanded)
    # The W vector is the sum of (coeff_C_block * s2_W_l_expanded)
    
    # Helper to expand a 2x2 vector (length 4) to a 4x4 vector (length 16)
    # Block 0 (00) corresponds to indices 0,1,4,5
    # Block 1 (01) corresponds to indices 2,3,6,7
    # Block 2 (10) corresponds to indices 8,9,12,13
    # Block 3 (11) corresponds to indices 10,11,14,15
    
    def expand_block(block_idx, vec, coeff):
        # vec is length 4: [a,b,c,d]
        # We return a dict {global_idx: val}
        res = {}
        # Map block_idx and local_idx to global_idx
        # Local 0 (a) -> r=0, c=0 -> global r*4+c
        # Local 1 (b) -> r=0, c=1
        # Local 2 (c) -> r=1, c=0
        # Local 3 (d) -> r=1, c=1
        
        # Block 0: r_base=0, c_base=0
        # Block 1: r_base=0, c_base=2
        # Block 2: r_base=2, c_base=0
        # Block 3: r_base=2, c_base=2
        
        r_base = (block_idx // 2) * 2
        c_base = (block_idx % 2) * 2
        
        for l in range(4):
            val = vec[l] * coeff
            if val != 0:
                r_local = l // 2
                c_local = l % 2
                r_global = r_base + r_local
                c_global = c_base + c_local
                global_idx = r_global * 4 + c_global
                res[global_idx] = res.get(global_idx, 0) + val
        return res
    
    col = 0
    for k in range(7):
        # Get linear combinations for A and B
        A_comb = outer_A[k]
        B_comb = outer_B[k]
        
        # The output C block is determined by the outer Strassen formula
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to know which C blocks this outer term M_k contributes to, and with what sign.
        # But the inner Strassen produces a 2x2 result for the block product.
        # So for each inner term l, we have a contribution to the 4 C blocks of the result of M_k.
        # Wait, the inner Strassen computes the product of two 2x2 matrices X and Y.
        # The result is a 2x2 matrix Z.
        # Z is added to the appropriate C blocks.
        
        # The inner Strassen terms s2_terms[l] give the U, V, W for the 2x2 product.
        # The W vector in s2_terms[l] tells us how the inner term contributes to the 2x2 result Z.
        # We need to map the 2x2 result Z to the 4x4 result C.
        
        # For outer term k, the result Z is added to C blocks with specific signs.
        # Let's define the signs for C blocks for each outer term k.
        # C00: M1(+), M4(+), M5(-), M7(+)
        # C01: M3(+), M5(+)
        # C10: M2(+), M4(+)
        # C11: M1(+), M2(-), M3(+), M6(+)
        
        c_signs = {
            0: {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7 contribute to C00
            1: {2: 1, 4: 1},               # M3, M5 contribute to C01
            2: {1: 1, 3: 1},               # M2, M4 contribute to C10
            3: {0: 1, 1: -1, 2: 1, 5: 1}   # M1, M2, M3, M6 contribute to C11
        }
        
        # For each inner term l:
        for l in range(7):
            u_vec, v_vec, w_vec = s2_terms[l]
            
            # Build U vector for this term
            u_dict = {}
            for block_idx, coeff in A_comb.items():
                expanded = expand_block(block_idx, u_vec, coeff)
                for idx, val in expanded.items():
                    u_dict[idx] = u_dict.get(idx, 0) + val
            
            # Build V vector for this term
            v_dict = {}
            for block_idx, coeff in B_comb.items():
                expanded = expand_block(block_idx, v_vec, coeff)
                for idx, val in expanded.items():
                    v_dict[idx] = v_dict.get(idx, 0) + val
            
            # Build W vector for this term
            # The inner term l contributes to the 2x2 result Z with weights w_vec.
            # w_vec is [z00, z01, z10, z11]
            # We need to map Z to C.
            # Z00 contributes to C blocks that M_k contributes to?
            # No, M_k is a 2x2 matrix product. Its result is a 2x2 matrix Z.
            # Z is added to C.
            # The mapping from Z to C is identity for the block structure?
            # Yes, if M_k is the product for C00, then Z00->C00, Z01->C01, etc.
            # But M_k is not just for one C block.
            # M1 contributes to C00 and C11.
            # So Z (result of M1) is added to C00 and C11.
            # Z00 -> C00[0,0], Z01 -> C00[0,1]? No.
            # C is 4x4. C00 is the top-left 2x2 block.
            # So Z00 -> C[0,0], Z01 -> C[0,1], Z10 -> C[1,0], Z11 -> C[1,1].
            # And if M1 also contributes to C11 (bottom-right), then:
            # Z00 -> C[2,2], Z01 -> C[2,3], Z10 -> C[3,2], Z11 -> C[3,3].
            
            # So for each inner term l, we add to W:
            # For each C block that M_k contributes to (with sign s):
            #   For each local index j in 0..3:
            #     global_idx = block_offset + j
            #     W[global_idx] += s * w_vec[j]
            
            w_dict = {}
            for c_block, s in c_signs.get(k, {}).items():
                # c_block is 0,1,2,3 corresponding to C00, C01, C10, C11
                # Wait, my c_signs dict above was keyed by C block index?
                # No, I defined it as {C_block_idx: {M_k_idx: sign}}?
                # Let's re-verify c_signs.
                # I wrote: c_signs[0] = {0: 1, 3: 1, 4: -1, 6: 1}
                # This means C block 0 (C00) is affected by M0, M3, M4, M6.
                # But M indices are 0..6.
                # M0 is M1. M3 is M4. M4 is M5. M6 is M7.
                # So C00 = M1 + M4 - M5 + M7. Correct.
                
                # So for a given outer term k (M_k), we need to find which C blocks it affects.
                # I can reverse the lookup.
                pass
            
            # Let's precompute the reverse mapping: for each M_k, list of (C_block, sign)
            # M0 (M1): C00(+), C11(+)
            # M1 (M2): C10(+), C11(-)
            # M2 (M3): C01(+), C11(+)
            # M3 (M4): C00(+), C10(+)
            # M4 (M5): C00(-), C01(+)
            # M5 (M6): C11(+)
            # M6 (M7): C00(+)
            
            m_to_c = [
                [(0, 1), (3, 1)],   # M0
                [(2, 1), (3, -1)],  # M1
                [(1, 1), (3, 1)],   # M2
                [(0, 1), (2, 1)],   # M3
                [(0, -1), (1, 1)],  # M4
                [(3, 1)],           # M5
                [(0, 1)]            # M6
            ]
            
            for c_block, sign in m_to_c[k]:
                # c_block is 0,1,2,3
                # Offset in 4x4 matrix:
                # C00: 0,1,4,5
                # C01: 2,3,6,7
                # C10: 8,9,12,13
                # C11: 10,11,14,15
                
                r_base = (c_block // 2) * 2
                c_base = (c_block % 2) * 2
                
                for j in range(4):
                    val = w_vec[j] * sign
                    if val != 0:
                        r_local = j // 2
                        c_local = j % 2
                        r_global = r_base + r_local
                        c_global = c_base + c_local
                        global_idx = r_global * 4 + c_global
                        w_dict[global_idx] = w_dict.get(global_idx, 0) + val
            
            # Now add to U, V, W matrices
            for idx, val in u_dict.items():
                U[idx][col] = val
            for idx, val in v_dict.items():
                V[idx][col] = val
            for idx, val in w_dict.items():
                W[idx][col] = val
            
            col += 1
    
    # 2. Add Schoolbook terms for 4x4x1
    # A(4x4) * B(4x1) -> C(4x1)
    # C_i = sum_j A[i,j] B[j]
    # B is 4x1, so B indices are 0..3.
    # C is 4x1, so C indices are 0..3.
    # Note: W matrix size is p*n = 5*4 = 20.
    # The Strassen part used W indices 0..15 (for the first 4 columns of C).
    # The Schoolbook part uses W indices 16..19 (for the 5th column of C).
    # Wait, C is 4x5.
    # Strassen part computes C[:, 0:4].
    # Schoolbook part computes C[:, 4].
    # So W indices for Schoolbook should be 4*4 + i = 16 + i?
    # W[k*n+i, r]. k is row of C (0..3), i is col of C (0..4).
    # So for C[:, 4], the index is k*4 + 4 = 4k + 4.
    # k=0 -> 4, k=1 -> 8, k=2 -> 12, k=3 -> 16.
    # Wait, the Strassen part used indices 0..15.
    # My Strassen W indices were 0..15.
    # Let's check: r_global*4 + c_global.
    # r_global 0..3, c_global 0..3.
    # Max index 3*4+3 = 15.
    # So Strassen uses W[0..15].
    # Schoolbook should use W[4, 8, 12, 16].
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j][col] = 1  # B is 4x1, so index j
            W[i * 4 + 4][col] = 1  # C is 4x5, col 4. Index i*4+4
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
