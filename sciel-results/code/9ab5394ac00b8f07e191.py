"""Rewritten program for 4x4 * 4x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    
    # We want to decompose the tensor T_{i,j,k} = delta_{i,j,k} for C = A B.
    # A is 4x4, B is 4x9, C is 4x9.
    # Indices:
    # A: i in [0,3], j in [0,3]. Index in U: i*m + j = i*4 + j.
    # B: j in [0,3], k in [0,8]. Index in V: j*p + k = j*9 + k.
    # C: k in [0,8], i in [0,3]. Index in W: k*n + i = k*4 + i. (Trace convention: W[k*n+i, r])
    
    # Strategy:
    # 1. Decompose M_{4,4,4} using recursive Strassen (Rank 49).
    # 2. Decompose M_{4,4,5} using schoolbook (Rank 80).
    # 3. Concatenate the factor matrices.
    
    # --- Part 1: Recursive Strassen for 4x4 * 4x4 ---
    # We define the Strassen decomposition for 2x2 matrices.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Products:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Coefficients for U (A part), V (B part), W (C part)
    # U indices for 2x2: 0:a, 1:b, 2:c, 3:d
    # V indices for 2x2: 0:e, 1:f, 2:g, 3:h
    # W indices for 2x2: 0:c00, 1:c01, 2:c10, 3:c11
    
    # m1: U=[1,0,1,0], V=[1,1,0,0], W=[1,0,0,1]
    # m2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # m3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    # m4: U=[0,0,0,1], V=[0,-1,1,0], W=[1,0,1,0]
    # m5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    # m6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # m7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    strassen_2x2 = [
        ([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Now, we use this to build the 4x4 * 4x4 decomposition.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each block is 2x2.
    # We apply the 7 Strassen products at the block level.
    # For each block-level product Mk = Uk * Vk, we expand Uk and Vk into scalar products using the 2x2 Strassen decomposition.
    
    # Block indices for 4x4 A (16 elements):
    # A11: 0,1,2,3
    # A12: 4,5,6,7
    # A21: 8,9,10,11
    # A22: 12,13,14,15
    
    # Block indices for 4x4 B (16 elements):
    # B11: 0,1,2,3
    # B12: 4,5,6,7
    # B21: 8,9,10,11
    # B22: 12,13,14,15
    
    # Block indices for 4x4 C (16 elements):
    # C11: 0,1,2,3
    # C12: 4,5,6,7
    # C21: 8,9,10,11
    # C22: 12,13,14,15
    
    # We will generate 7 * 7 = 49 scalar products.
    # U_list, V_list, W_list for the 4x4 part.
    U_44 = []
    V_44 = []
    W_44 = []
    
    # Define the 7 block-level Strassen terms for 4x4.
    # Each term is (U_block_coeffs, V_block_coeffs, W_block_coeffs)
    # U_block_coeffs is a 4x1 vector of coefficients for A11, A12, A21, A22.
    # V_block_coeffs is a 4x1 vector of coefficients for B11, B12, B21, B22.
    # W_block_coeffs is a 4x1 vector of coefficients for C11, C12, C21, C22.
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    block_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Helper to expand a 2x2 Strassen decomposition into scalar products
    # given the block coefficients.
    # u_block: [u11, u12, u21, u22] coefficients for the 2x2 matrix U.
    # v_block: [v11, v12, v21, v22] coefficients for the 2x2 matrix V.
    # w_block: [w11, w12, w21, w22] coefficients for the 2x2 matrix W.
    # base_u_offset: offset for U indices in the 4x4 matrix.
    # base_v_offset: offset for V indices in the 4x4 matrix.
    # base_w_offset: offset for W indices in the 4x4 matrix.
    
    # We need to map the 2x2 indices (0,1,2,3) to the 4x4 indices.
    # For A (U):
    # A11 (block 0) -> indices 0,1,2,3
    # A12 (block 1) -> indices 4,5,6,7
    # A21 (block 2) -> indices 8,9,10,11
    # A22 (block 3) -> indices 12,13,14,15
    
    # For B (V):
    # B11 (block 0) -> indices 0,1,2,3
    # B12 (block 1) -> indices 4,5,6,7
    # B21 (block 2) -> indices 8,9,10,11
    # B22 (block 3) -> indices 12,13,14,15
    
    # For C (W):
    # C11 (block 0) -> indices 0,1,2,3
    # C12 (block 1) -> indices 4,5,6,7
    # C21 (block 2) -> indices 8,9,10,11
    # C22 (block 3) -> indices 12,13,14,15
    
    # Wait, the indices for U in the final matrix are i*4+j.
    # A11 is i in 0,1 and j in 0,1. Indices: 0,1,4,5.
    # A12 is i in 0,1 and j in 2,3. Indices: 2,3,6,7.
    # A21 is i in 2,3 and j in 0,1. Indices: 8,9,12,13.
    # A22 is i in 2,3 and j in 2,3. Indices: 10,11,14,15.
    
    # Let's define the mapping from block-local 2x2 indices (0,1,2,3) to global 4x4 indices.
    # Local 2x2:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (1,0)
    # 3: (1,1)
    
    # Block 0 (A11):
    # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # Block 1 (A12):
    # (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # Block 2 (A21):
    # (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # Block 3 (A22):
    # (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    u_block_maps = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Same mapping for V (B) and W (C) since dimensions are 4x4.
    v_block_maps = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    w_block_maps = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Iterate over the 7 block-level Strassen terms
    for b_u, b_v, b_w in block_strassen:
        # For each term, we have a product of two 2x2 matrices.
        # We expand this into 7 scalar products using the 2x2 Strassen decomposition.
        for s_u, s_v, s_w in strassen_2x2:
            # s_u is coefficients for the local 2x2 U matrix.
            # b_u is coefficients for the 4 2x2 blocks of U.
            # The overall coefficient for a scalar element is the sum of products of block coeff and local coeff.
            # Actually, it's simpler:
            # U_scalar = sum_{b=0}^3 b_u[b] * U_local_b
            # where U_local_b is the 2x2 matrix corresponding to block b.
            # And U_local_b is formed by the linear combination defined by s_u.
            
            # Let's construct the full 16-dim vectors for U, V, W.
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16
            
            # For each block b (0..3):
            for b in range(4):
                coeff_b_u = b_u[b]
                coeff_b_v = b_v[b]
                coeff_b_w = b_w[b]
                
                if coeff_b_u == 0 and coeff_b_v == 0 and coeff_b_w == 0:
                    continue
                
                # For this block, the local 2x2 U matrix is determined by s_u.
                # The local 2x2 V matrix is determined by s_v.
                # The local 2x2 W matrix is determined by s_w.
                
                # The contribution to U_vec is coeff_b_u * s_u mapped to block b.
                # The contribution to V_vec is coeff_b_v * s_v mapped to block b.
                # The contribution to W_vec is coeff_b_w * s_w mapped to block b.
                
                # Wait, this logic is slightly off.
                # The Strassen formula says:
                # C_block = sum_{b} W_b * (U_b * V_b)
                # Where U_b and V_b are linear combinations of A_blocks and B_blocks.
                # My `block_strassen` defines U_b and V_b as linear combinations of A_blocks.
                # So U_b is a 2x2 matrix.
                # Then U_b * V_b is a product of two 2x2 matrices.
                # We decompose U_b * V_b into 7 scalar products using `strassen_2x2`.
                # The `strassen_2x2` entries `s_u, s_v, s_w` define the scalar products for a generic 2x2 * 2x2.
                # So, for a specific block b, the scalar products are formed by:
                # U_local = s_u (applied to the elements of U_b)
                # V_local = s_v (applied to the elements of V_b)
                # W_local = s_w (applied to the elements of C_b)
                
                # However, U_b itself is a linear combination of A_blocks.
                # This means we are effectively doing a tensor product of the decompositions.
                
                # Let's re-verify the composition.
                # We have 7 block products M_k = U_k A * V_k B.
                # Each M_k is a 2x2 matrix.
                # We want to compute C = sum_k W_k M_k.
                # We decompose M_k into 7 scalar products.
                # So for each k, we have 7 scalar products.
                # Total 49.
                
                # For a fixed k (block term) and fixed l (local Strassen term):
                # The scalar product is (U_k A)_{s_u} * (V_k B)_{s_v}
                # where (M)_{s_u} means the linear combination of M's elements defined by s_u.
                
                # So, U_global = sum_{b} b_u[b] * (A_block_b)_{s_u}
                # This is a linear combination of A's elements.
                
                # Let's compute the 16-dim vectors.
                for local_idx in range(4):
                    # local_idx 0,1,2,3 correspond to 2x2 positions (0,0), (0,1), (1,0), (1,1)
                    u_coeff_local = s_u[local_idx]
                    v_coeff_local = s_v[local_idx]
                    w_coeff_local = s_w[local_idx]
                    
                    # For each block b:
                    for b in range(4):
                        u_coeff_block = b_u[b]
                        v_coeff_block = b_v[b]
                        w_coeff_block = b_w[b]
                        
                        u_val = u_coeff_block * u_coeff_local
                        v_val = v_coeff_block * v_coeff_local
                        w_val = w_coeff_block * w_coeff_local
                        
                        # Map local_idx and block b to global index
                        u_global_idx = u_block_maps[b][local_idx]
                        v_global_idx = v_block_maps[b][local_idx]
                        w_global_idx = w_block_maps[b][local_idx]
                        
                        u_vec[u_global_idx] += u_val
                        v_vec[v_global_idx] += v_val
                        w_vec[w_global_idx] += w_val
            
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
            
    # --- Part 2: Schoolbook for 4x4 * 4x5 ---
    # We need to decompose M_{4,4,5}.
    # A is 4x4, B is 4x5, C is 4x5.
    # Schoolbook rank = 4*4*5 = 80.
    
    # We will append these to the previous lists.
    # But we need to be careful with indices.
    # U indices are the same (0..15).
    # V indices for the 4x5 part: j in 0..3, k in 0..4.
    # In the full 4x9 matrix, V indices are j*9+k.
    # For the 4x4 part (columns 0..3 of B), V indices were j*4+k? No, V is (m*p, R).
    # V shape is (4*9, R).
    # For the 4x4 part, we used V indices corresponding to B being 4x4.
    # But B is 4x9.
    # The V matrix for the 4x4 part should have non-zeros only in the first 16 columns of B (if we viewed B as 4x4).
    # But B is 4x9. The first 4 columns of B are B[:, 0:4].
    # So for the 4x4 part, V indices should be j*9 + k for k in 0..3.
    
    # Let's re-evaluate the V indices for the 4x4 part.
    # In my `v_block_maps`, I used indices 0..15.
    # These correspond to a 4x4 matrix.
    # In the full 4x9 matrix, the elements of B are at indices j*9 + k.
    # For the first 4 columns (k=0..3), the indices are:
    # j=0: 0, 1, 2, 3
    # j=1: 9, 10, 11, 12
    # j=2: 18, 19, 20, 21
    # j=3: 27, 28, 29, 30
    
    # My `v_block_maps` assumed a contiguous 4x4 layout (0..15).
    # I need to remap the V indices for the 4x4 part to the correct positions in the 4x9 matrix.
    
    # Similarly for W.
    # W is (p*n, R) = (9*4, R) = (36, R).
    # W indices are k*4 + i.
    # For the 4x4 part (C is 4x4, k=0..3, i=0..3), the indices are:
    # k=0: 0, 1, 2, 3
    # k=1: 4, 5, 6, 7
    # k=2: 8, 9, 10, 11
    # k=3: 12, 13, 14, 15
    
    # My `w_block_maps` assumed a contiguous 4x4 layout (0..15).
    # This matches the W indices for k=0..3 in the 4x9 case?
    # W index = k*4 + i.
    # For k=0, i=0..3 -> 0,1,2,3.
    # For k=1, i=0..3 -> 4,5,6,7.
    # Yes, the first 16 entries of W (for k=0..3) are contiguous 0..15.
    # So W indices are fine.
    
    # But V indices are NOT contiguous.
    # V index = j*9 + k.
    # My `v_block_maps` gave 0..15.
    # I need to transform the V vectors from the 4x4 part.
    
    # Let's create a mapping from "local 4x4 V index" to "global 4x9 V index".
    # Local index l (0..15) corresponds to j = l // 4, k = l % 4.
    # Global index = j*9 + k.
    
    v_map_4x4_to_4x9 = []
    for l in range(16):
        j = l // 4
        k = l % 4
        v_map_4x4_to_4x9.append(j * 9 + k)
        
    # Now, we adjust the V_44 vectors.
    # Original V_44 vectors have length 16.
    # We need to expand them to length 36 (4*9) and place the values at the correct indices.
    
    V_44_adjusted = []
    for vec in V_44:
        new_vec = [0.0] * 36
        for l in range(16):
            if vec[l] != 0:
                new_vec[v_map_4x4_to_4x9[l]] = vec[l]
        V_44_adjusted.append(new_vec)
        
    # U_44 vectors are length 16. This is correct for U (4*4=16).
    # W_44 vectors are length 16. We need to expand them to length 36 (9*4=36).
    # The non-zeros are in the first 16 entries (k=0..3).
    # So we just pad with zeros.
    
    W_44_adjusted = []
    for vec in W_44:
        new_vec = vec + [0.0] * 20
        W_44_adjusted.append(new_vec)
        
    # Now, let's build the schoolbook part for 4x4 * 4x5.
    # This covers columns 4..8 of B.
    # B indices: j*9 + k for j in 0..3, k in 4..8.
    # C indices: k*4 + i for k in 4..8, i in 0..3.
    
    U_school = []
    V_school = []
    W_school = []
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                u_vec = [0.0] * 16
                u_vec[i * 4 + j] = 1.0
                
                v_vec = [0.0] * 36
                v_vec[j * 9 + k] = 1.0
                
                w_vec = [0.0] * 36
                w_vec[k * 4 + i] = 1.0
                
                U_school.append(u_vec)
                V_school.append(v_vec)
                W_school.append(w_vec)
                
    # Combine all parts
    U_final = U_44 + U_school
    V_final = V_44_adjusted + V_school
    W_final = W_44_adjusted + W_school
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
