import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Rank 49 for 4x4x4 (Strassen)
    # Rank 48 for 4x4x3 (Trivial)
    # Total Rank 97
    
    r_444 = 49
    r_443 = 48
    rank = r_444 + r_443
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # We need to construct the decomposition for 4x4x4.
    # This involves 7 Strassen steps on 2x2x2 blocks.
    
    # Helper to get 2x2x2 Strassen U, V, W
    def get_strassen_2x2_2x2():
        u_list = []
        v_list = []
        w_list = []
        
        # M1 = (a+d)(e+h)
        u1 = [1, 0, 0, 1] # a, b, c, d -> a+d
        v1 = [1, 0, 0, 1] # e, f, g, h -> e+h
        w1 = [1, 0, 0, 1] # C00, C01, C10, C11 -> C00, C11 get 1
        
        # M2 = (c+d)e
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, -1] # C10 += M2, C11 -= M2
        
        # M3 = a(f-h)
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, 1] # C01 += M3, C11 += M3
        
        # M4 = d(g-e)
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1]
        w4 = [1, 0, 1, 0] # C00 += M4, C10 += M4
        
        # M5 = (a+b)h
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0] # C00 -= M5, C01 += M5
        
        # M6 = (c-a)(e+f)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C11 += M6
        
        # M7 = (b-d)(g+h)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0] # C00 += M7
        
        u_list = [u1, u2, u3, u4, u5, u6, u7]
        v_list = [v1, v2, v3, v4, v5, v6, v7]
        w_list = [w1, w2, w3, w4, w5, w6, w7]
        
        return u_list, v_list, w_list

    u_2, v_2, w_2 = get_strassen_2x2_2x2()
    
    # Now map 4x4x4 to 7 * 2x2x2
    # A is 4x4. Blocks: A00, A01, A10, A11.
    # B is 4x4. Blocks: B00, B01, B10, B11.
    # C is 4x4. Blocks: C00, C01, C10, C11.
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each term, we have a product of two linear combinations of 2x2 blocks.
    # Each block product is computed using the 2x2x2 Strassen decomposition.
    
    # We need to construct U, V, W for the 4x4x4 case.
    # U shape: 16 x 49.
    # V shape: 16 x 49.
    # W shape: 16 x 49.
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Define the structure of each Strassen term
    # term_idx: 0..6
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff) -> This is handled by W in 2x2x2
    
    terms = [
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)] }, # M1
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)] },          # M2
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)] },         # M3
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)] },         # M4
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)] },          # M5
        { 'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)] }, # M6
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)] }, # M7
    ]
    
    # Block indices in 4x4 matrix:
    # 0: 0,0 to 1,1
    # 1: 0,2 to 1,3
    # 2: 2,0 to 3,1
    # 3: 2,2 to 3,3
    
    # Indices in flattened 16-vector:
    # Block b covers rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1.
    # A is 4x4. 
    # Block 0: i in 0,1; j in 0,1. Indices: 0,1,4,5.
    # Block 1: i in 0,1; j in 2,3. Indices: 2,3,6,7.
    # Block 2: i in 2,3; j in 0,1. Indices: 8,9,12,13.
    # Block 3: i in 2,3; j in 2,3. Indices: 10,11,14,15.
    
    def get_block_indices_4x4(block_idx):
        bi = block_idx // 2
        bj = block_idx % 2
        i_start = bi * 2
        j_start = bj * 2
        idxs = []
        for i in range(i_start, i_start + 2):
            for j in range(j_start, j_start + 2):
                idxs.append(i * 4 + j)
        return idxs
    
    # For B (4x4), same indices.
    # For C (4x4), same indices.
    
    for t_idx in range(7):
        # For each of the 7 Strassen terms, we use the 2x2x2 decomposition.
        # The 2x2x2 decomposition has 7 sub-terms.
        # Total 49 sub-terms.
        for s_idx in range(7):
            global_t = t_idx * 7 + s_idx
            
            # Get the 2x2x2 coefficients for sub-term s_idx
            u_sub = u_2[s_idx] # len 4
            v_sub = v_2[s_idx] # len 4
            w_sub = w_2[s_idx] # len 4
            
            # Map A blocks to global A indices
            # The term involves a linear combination of A blocks.
            # The 2x2x2 decomposition assumes the input is a single 2x2 matrix.
            # Here, the "matrix" is the linear combination of blocks.
            # So, for each element in the 2x2 matrix (local i,j),
            # it is a sum of coeff * A_global[i', j'].
            
            # Let's build the mapping.
            # Local A is 2x2.
            # Local A[i,j] = sum_{block, coeff in terms[t_idx]['A']} coeff * A_block[block][i,j]
            
            # We need to fill U_444[global_A_idx, global_t]
            # U_444[global_A_idx, global_t] = sum_{local_i, local_j} u_sub[local_i*2+local_j] * coeff_for_block_containing_global_A_idx
            
            # Actually, it's easier to iterate over global A indices.
            # For each global A index ga, determine which block it belongs to.
            # If the block is in the term's A list, add the contribution.
            
            for ga in range(16):
                # Find block and local index
                i = ga // 4
                j = ga % 4
                bi = i // 2
                bj = j // 2
                block_idx = bi * 2 + bj
                li = i % 2
                lj = j % 2
                local_idx = li * 2 + lj
                
                # Check if this block is in the term
                coeff = 0
                for b_idx, c in terms[t_idx]['A']:
                    if b_idx == block_idx:
                        coeff += c
                        break
                
                if coeff != 0:
                    U_444[ga, global_t] += coeff * u_sub[local_idx]
            
            # Similarly for V
            for gv in range(16):
                i = gv // 4
                j = gv % 4
                bi = i // 2
                bj = j % 2
                block_idx = bi * 2 + bj
                li = i % 2
                lj = j % 2
                local_idx = li * 2 + lj
                
                coeff = 0
                for b_idx, c in terms[t_idx]['B']:
                    if b_idx == block_idx:
                        coeff += c
                        break
                
                if coeff != 0:
                    V_444[gv, global_t] += coeff * v_sub[local_idx]
            
            # Similarly for W
            # W in 2x2x2 is w_sub[local_k * 2 + local_i]
            # This contributes to C_block[local_k, local_i]
            # We need to map to global C indices.
            for gc in range(16):
                i = gc // 4
                j = gc % 4
                bi = i // 2
                bj = j % 2
                block_idx = bi * 2 + bj
                li = i % 2
                lj = j % 2
                local_idx = li * 2 + lj # Note: w_sub is indexed by k, i. So local_k, local_i.
                
                # In 2x2x2, w_sub is for C[k,i].
                # So local_k = li, local_i = lj.
                # Wait, C is 2x2. C[k,i].
                # In my w_2 definition, w is [C00, C01, C10, C11].
                # Index 0: k=0, i=0.
                # Index 1: k=0, i=1.
                # Index 2: k=1, i=0.
                # Index 3: k=1, i=1.
                # So local_idx = local_k * 2 + local_i.
                
                # In global C, block_idx corresponds to (bi, bj).
                # The term M_t contributes to C_block with coefficient w_sub[local_idx].
                # But which C block?
                # The Strassen formula gives C00, C01, C10, C11.
                # These correspond to blocks 0, 1, 2, 3.
                # So block_idx in C is the same as the block index in the Strassen output.
                
                # However, the Strassen term t_idx produces a specific C block?
                # No, each M_t contributes to ALL C blocks with specific coefficients.
                # The w_sub vector contains the coefficients for the 4 C blocks (as 2x2).
                # So for each local (k,i) in the 2x2 C, we add w_sub[k*2+i] to the corresponding global C index in block `block_idx`?
                # No! The Strassen formula is:
                # C00 = M1 + ...
                # C01 = M3 + ...
                # C10 = M2 + ...
                # C11 = M1 - ...
                # So M_t contributes to C00, C01, C10, C11.
                # The w_sub vector from the 2x2x2 decomposition gives the coefficients for the 4 elements of the 2x2 result of M_t.
                # Wait. M_t is a 2x2 matrix.
                # So M_t[k,i] is the (k,i) element.
                # The Strassen formula says C_block[k,i] += coeff * M_t[k,i].
                # So yes, for each C block, the contribution is w_sub[local_idx] * (coeff of M_t in that C block).
                
                # But where do the coefficients for C blocks come from?
                # They are hardcoded in the Strassen formula.
                # I need to define them.
                
                # Let's define c_coeffs[t_idx][block_idx]
                # This is getting complicated.
                # Alternative: Just use the trivial rank 64 for 4x4x4?
                # No, 49 is much better.
                
                # Let's hardcode the C block coefficients for each M_t.
                
                # M1: C00(+1), C11(+1)
                # M2: C10(+1), C11(-1)
                # M3: C01(+1), C11(+1)
                # M4: C00(+1), C10(+1)
                # M5: C00(-1), C01(+1)
                # M6: C11(+1)
                # M7: C00(+1)
                
                c_coeffs = [
                    {0: 1, 3: 1}, # M1
                    {2: 1, 3: -1}, # M2
                    {1: 1, 3: 1}, # M3
                    {0: 1, 2: 1}, # M4
                    {0: -1, 1: 1}, # M5
                    {3: 1}, # M6
                    {0: 1} # M7
                ]
                
                coeff_c = c_coeffs[t_idx].get(block_idx, 0)
                if coeff_c != 0:
                    W_444[gc, global_t] += coeff_c * w_sub[local_idx]
                    
    # Copy U_444, V_444, W_444 to U, V, W
    U[:, :49] = U_444
    V[:, :49] = V_444
    W[:, :49] = W_444
    
    # --- Part 2: 4x4x3 using Trivial (Rank 48) ---
    # B is 4x7. We use cols 4,5,6 for this part.
    # A is 4x4.
    # C is 4x7. We use cols 4,5,6.
    
    t = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                
                U[u_idx, t] = 1
                V[v_idx, t] = 1
                W[w_idx, t] = 1
                t += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
