# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 6
    
    # Dimensions for block structure
    # A is 4x4: 2x2 blocks of 2x2
    # B is 4x6: 2x3 blocks of 2x2
    # C is 4x6: 2x3 blocks of 2x2
    
    # Inner rank for 2x2 * 2x2 -> 2x2 is 7 (Strassen)
    # Outer rank for 2x2 * 2x3 -> 2x3 is 11
    
    # We need to construct the full U, V, W matrices of sizes:
    # U: (n*m, rank) = (16, 77)
    # V: (m*p, rank) = (24, 77)
    # W: (p*n, rank) = (24, 77)
    
    # 1. Define Strassen coefficients for 2x2 * 2x2 -> 2x2
    # U_inner: (4, 7) where row = i*2+j
    # V_inner: (4, 7) where row = j*2+k
    # W_inner: (4, 7) where row = k*2+i (Trace convention: W[k*n+i])
    
    U_in = np.zeros((4, 7), dtype=int)
    V_in = np.zeros((4, 7), dtype=int)
    W_in = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U_in[0, 0] = 1; U_in[3, 0] = 1
    V_in[0, 0] = 1; V_in[3, 0] = 1
    W_in[0, 0] = 1  # C00
    W_in[3, 0] = 1  # C11
    
    # M2 = (A10 + A11) * B00
    U_in[2, 1] = 1; U_in[3, 1] = 1
    V_in[0, 1] = 1
    W_in[1, 1] = 1  # C10
    W_in[3, 1] = -1 # C11
    
    # M3 = A00 * (B01 - B11)
    U_in[0, 2] = 1
    V_in[1, 2] = 1; V_in[3, 2] = -1
    W_in[2, 2] = 1  # C01
    W_in[3, 2] = 1  # C11
    
    # M4 = A11 * (B10 - B00)
    U_in[3, 3] = 1
    V_in[2, 3] = 1; V_in[0, 3] = -1
    W_in[0, 3] = 1  # C00
    W_in[1, 3] = 1  # C10
    
    # M5 = (A00 + A01) * B11
    U_in[0, 4] = 1; U_in[1, 4] = 1
    V_in[3, 4] = 1
    W_in[0, 4] = -1 # C00
    W_in[2, 4] = 1  # C01
    
    # M6 = (A10 - A00) * (B00 + B01)
    U_in[2, 5] = 1; U_in[0, 5] = -1
    V_in[0, 5] = 1; V_in[1, 5] = 1
    W_in[3, 5] = 1  # C11
    
    # M7 = (A01 - A11) * (B10 + B11)
    U_in[1, 6] = 1; U_in[3, 6] = -1
    V_in[2, 6] = 1; V_in[3, 6] = 1
    W_in[0, 6] = 1  # C00
    W_in[3, 6] = 1  # C11
    
    # 2. Define Outer decomposition for 2x2 * 2x3 -> 2x3
    # This involves 11 terms.
    # We can construct this by combining:
    # - 7 terms for the first 2 columns (using Strassen on the 2x2 * 2x2 part)
    # - 4 terms for the 3rd column (using schoolbook on the 2x2 * 2x1 part)
    
    # The outer terms map:
    # U_outer[ai, bj] for A block (ai, bj)
    # V_outer[bj, ck] for B block (bj, ck)
    # W_outer[ai, ck] for C block (ai, ck)
    
    # We will iterate through the 11 outer terms and 7 inner terms.
    # Total rank = 11 * 7 = 77.
    
    U = np.zeros((16, 77), dtype=int)
    V = np.zeros((24, 77), dtype=int)
    W = np.zeros((24, 77), dtype=int)
    
    # Helper to add a term to U, V, W
    # outer_idx: 0..10
    # inner_idx: 0..6
    # col = outer_idx * 7 + inner_idx
    
    # We need to define the 11 outer terms.
    # Term 0-6: Strassen for columns 0,1.
    # For each Strassen term t (0..6):
    #   U_block[ai, bj] = U_in[ai*2+ai2, t] ... wait.
    #   The outer term corresponds to a specific combination of block indices.
    
    # Let's generate the outer terms explicitly.
    # Strassen for 2x2 * 2x2 (cols 0,1):
    # The U_in, V_in, W_in above are for the INNER 2x2 multiplication.
    # The OUTER multiplication is between the 2x2 block matrices.
    # Let A_blocks be 2x2, B_blocks be 2x3.
    # C_blocks[ai, ck] = sum_bj A_blocks[ai, bj] * B_blocks[bj, ck].
    
    # For cols 0,1 (ck=0,1), we use Strassen on the 2x2 * 2x2 part.
    # The "inner" Strassen (U_in, etc.) acts on the elements WITHIN the 2x2 blocks.
    # Wait, this is a Kronecker product.
    # The tensor is T = T_outer \otimes T_inner.
    # T_outer is the tensor for 2x2 * 2x3 -> 2x3 block multiplication.
    # T_inner is the tensor for 2x2 * 2x2 -> 2x2 element multiplication.
    
    # The rank is Rank(T_outer) * Rank(T_inner) = 11 * 7 = 77.
    
    # We need the explicit coefficients for T_outer.
    # T_outer has dimensions (2*2) x (2*3) x (2*3) = 4 x 6 x 6.
    # But we only need the non-zero patterns for the 11 terms.
    
    # Let's define the 11 outer terms as lists of (ai, bj), (bj, ck), (ai, ck) with coeff 1.
    # Actually, since it's a sum of rank-1 tensors, each term is a product of vectors.
    # u_vec: length 4 (2x2 blocks of A)
    # v_vec: length 6 (2x3 blocks of B)
    # w_vec: length 6 (2x3 blocks of C)
    
    # Terms 0-6 come from Strassen applied to the 2x2 * 2x2 sub-problem (cols 0,1).
    # Strassen for 2x2 * 2x2:
    # M1: u=(1,0,0,1), v=(1,0,0,0), w=(1,0,0,1) ? No, w is for C.
    # Let's use the U_in, V_in, W_in from the inner Strassen as the OUTER coefficients for the first 7 terms?
    # NO. The inner Strassen is for the 2x2 elements.
    # The outer Strassen is for the 2x2 blocks.
    # The coefficients are the SAME structure!
    # So for the first 7 outer terms, the block coefficients are the same as the inner Strassen coefficients.
    
    outer_terms = []
    
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        
        # Extract non-zero entries from U_in, V_in, W_in
        for ai in range(2):
            for bj in range(2):
                c = U_in[ai*2+bj, t]
                if c != 0:
                    u_dict[(ai, bj)] = c
        for bj in range(2):
            for ck in range(2): # Only ck 0,1 for this part
                c = V_in[bj*2+ck, t]
                if c != 0:
                    v_dict[(bj, ck)] = c
        for ai in range(2):
            for ck in range(2): # Only ck 0,1 for this part
                c = W_in[ck*2+ai, t]
                if c != 0:
                    w_dict[(ai, ck)] = c
                    
        outer_terms.append((u_dict, v_dict, w_dict))
        
    # 4 Schoolbook terms for col 2
    # C[ai, 2] = A[ai, 0]*B[0, 2] + A[ai, 1]*B[1, 2]
    for ai in range(2):
        for bj in range(2):
            u_dict = {(ai, bj): 1}
            v_dict = {(bj, 2): 1}
            w_dict = {(ai, 2): 1}
            outer_terms.append((u_dict, v_dict, w_dict))
            
    # Now combine
    for ot_idx, (u_dict, v_dict, w_dict) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part
            for (ai, bj), c_out in u_dict.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        c_in = U_in[ai2*2 + bj2, t]
                        if c_out != 0 and c_in != 0:
                            u_row = (ai*2 + ai2) * 4 + (bj*2 + bj2)
                            U[u_row, col] += c_out * c_in
                            
            # V part
            for (bj, ck), c_out in v_dict.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        c_in = V_in[bj2*2 + ck2, t]
                        if c_out != 0 and c_in != 0:
                            v_row = (bj*2 + bj2) * 6 + (ck*2 + ck2)
                            V[v_row, col] += c_out * c_in
                            
            # W part
            for (ai, ck), c_out in w_dict.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        c_in = W_in[ck2*2 + ai2, t]
                        if c_out != 0 and c_in != 0:
                            w_row = (ck*2 + ck2) * 4 + (ai*2 + ai2)
                            W[w_row, col] += c_out * c_in

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
