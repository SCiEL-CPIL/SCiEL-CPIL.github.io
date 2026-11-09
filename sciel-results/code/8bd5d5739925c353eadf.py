import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W as zero matrices
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 2x2 patterns
    # u, v, w are vectors of length 4 (flattened 2x2 matrices)
    # Indexing: 00, 01, 10, 11
    # u[k] corresponds to A[k//2, k%2]
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # C00 += M1
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10 += M2
        # M3 = A00(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # C01 += M3
        # M4 = A11(B10-B00)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # C00 += M4
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # C00 -= M5
        # M6 = (A10-A00)(B00+B01)
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]), # C00 += M7
    ]
    
    # Wait, the standard Strassen reconstruction is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My list above has w vectors that are basis vectors for C entries.
    # I need to accumulate them correctly.
    # Let's redefine the list to include the target C entry and sign.
    
    # Format: (u, v, w_target, sign)
    # w_target is index in {0,1,2,3} for C00, C01, C10, C11
    # u, v are length 4 vectors
    
    patterns = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], 0, 1),  # C00 += M1
        ([1, 0, 0, 1], [1, 0, 0, 1], 3, 1),  # C11 += M1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], 2, 1),  # C10 += M2
        ([0, 0, 1, 1], [1, 0, 0, 0], 3, -1), # C11 -= M2
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], 1, 1), # C01 += M3
        ([1, 0, 0, 0], [0, 1, 0, -1], 3, 1), # C11 += M3
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], 0, 1), # C00 += M4
        ([0, 0, 0, 1], [0, 0, 1, -1], 2, 1), # C10 += M4
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], 0, -1), # C00 -= M5
        ([1, 1, 0, 0], [0, 0, 0, 1], 1, 1),  # C01 += M5
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], 3, 1), # C11 += M6
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], 0, 1), # C00 += M7
    ]
    
    # Top-level Strassen for 4x4
    # Blocks: 00, 01, 10, 11
    # X_t = sum c^X_{t, ij} A_{ij}
    # Y_t = sum c^Y_{t, ij} B_{ij}
    
    top_patterns = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    col = 0
    for t_idx, (X_coeffs, Y_coeffs) in enumerate(top_patterns):
        for u, v, w_target, sign in patterns:
            # u, v are length 4 vectors for the 2x2 multiplication of X_t and Y_t
            # w_target is the entry of P_t (2x2 matrix) that receives this term
            # sign is the coefficient for this term in the final C assembly
            
            # Construct U vector for this term
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # Map u, v to global indices
            # u[k] is coeff for k-th entry of X_t
            # X_t[k] = sum_{ij} X_coeffs.get((i,j), 0) * A_{ij}[k]
            # A_{ij}[k] is at global index (i*2 + k//2)*4 + (j*2 + k%2)
            
            for k in range(4):
                if u[k] != 0:
                    for (i, j), c in X_coeffs.items():
                        if c != 0:
                            # Global index for A_{ij}[k]
                            # i, j in {0,1}
                            # k in {0,1,2,3} -> r=k//2, c=k%2
                            r = k // 2
                            c_loc = k % 2
                            g_row = i * 2 + r
                            g_col = j * 2 + c_loc
                            g_idx = g_row * 4 + g_col
                            U_vec[g_idx] += u[k] * c
                
                if v[k] != 0:
                    for (i, j), c in Y_coeffs.items():
                        if c != 0:
                            r = k // 2
                            c_loc = k % 2
                            g_row = i * 2 + r
                            g_col = j * 2 + c_loc
                            g_idx = g_row * 4 + g_col
                            V_vec[g_idx] += v[k] * c
            
            # W vector
            # w_target is index in {0,1,2,3} for P_t entries
            # P_t is a 2x2 matrix.
            # The term contributes to C with sign.
            # The final C is assembled from P_t's.
            # C = sum_t D_t P_t
            # where D_t are matrices of coefficients.
            # Actually, the patterns list already accounts for the distribution of P_t entries to C.
            # Wait, the patterns list is for the 2x2 multiplication P_t = X_t Y_t.
            # The output of that multiplication is P_t, a 2x2 matrix.
            # The 'w_target' in patterns refers to the entry of P_t.
            # But we need to map P_t entries to C entries.
            
            # Let's re-evaluate.
            # The Strassen algorithm computes C directly from M1..M7.
            # My 'patterns' list decomposes the 2x2 multiplication tensor.
            # So for each top-level term t, we compute P_t = X_t Y_t.
            # The decomposition of P_t = X_t Y_t gives terms that contribute to P_t entries.
            # Then we need to add these contributions to C.
            
            # The mapping from P_t entries to C entries depends on t.
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # So for a given t and a given pattern (u, v, w_target, sign):
            # The term contributes to P_t[w_target].
            # This contribution is then added to C with a coefficient depending on t and w_target.
            
            # Let's define the assembly coefficients for each t.
            # assembly[t][w_target] = coeff in C
            # t is 0..6 (M1..M7)
            # w_target is 0..3 (P00, P01, P10, P11)
            
            assembly = [
                # M1
                {0: 1, 1: 0, 2: 0, 3: 1}, # C00 += P1, C11 += P1
                # M2
                {0: 0, 1: 0, 2: 1, 3: -1}, # C10 += P2, C11 -= P2
                # M3
                {0: 0, 1: 1, 2: 0, 3: 1},  # C01 += P3, C11 += P3
                # M4
                {0: 1, 1: 0, 2: 1, 3: 0},  # C00 += P4, C10 += P4
                # M5
                {0: -1, 1: 1, 2: 0, 3: 0}, # C00 -= P5, C01 += P5
                # M6
                {0: 0, 1: 0, 2: 0, 3: 1},  # C11 += P6
                # M7
                {0: 1, 1: 0, 2: 0, 3: 0},  # C00 += P7
            ]
            
            coeff_in_C = assembly[t_idx].get(w_target, 0)
            if coeff_in_C == 0:
                continue # This term doesn't contribute to any C entry? No, w_target is fixed.
            
            # The sign from the pattern is already included in u, v?
            # No, u, v are the vectors for the multiplication.
            # The 'sign' in patterns was intended for the C assembly, but I put it in patterns.
            # Let's check patterns again.
            # I defined patterns with w_target and sign.
            # But the assembly logic above is separate.
            # The patterns list I defined earlier had w_target and sign.
            # Let's remove 'sign' from patterns and use only assembly.
            # Actually, the patterns list I wrote above has 13 entries?
            # No, Strassen has 7 multiplications.
            # My patterns list has 13 entries because I expanded the reconstruction?
            # No, the patterns list should have 7 entries, one for each M_t.
            # And each M_t contributes to multiple C entries.
            # The tensor decomposition is a sum of rank-1 tensors.
            # Each rank-1 tensor is u \otimes v \otimes w.
            # Here w is a vector of length 16.
            # The w vector for a specific multiplication M_t and a specific output entry C_k
            # is determined by the assembly.
            
            # Let's restart the inner loop logic.
            # For each top-level term t (0..6):
            #   For each bottom-level term s (0..6):
            #     u_s, v_s are vectors of length 4.
            #     This term computes a scalar product that contributes to P_t.
            #     Specifically, it contributes to all entries of P_t?
            #     No, the 2x2 multiplication tensor is T = sum_s u_s \otimes v_s \otimes w_s.
            #     w_s is a vector of length 4 (entries of P).
            #     So for each s, we have a w_s vector.
            #     We need to map w_s to the global C vector.
            
            # Let's redefine patterns to be the 7 Strassen terms with their w vectors.
            
            pass # Placeholder

    # Redoing the construction properly
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # 2x2 Strassen terms: (u, v, w)
    # u, v, w are length 4 vectors
    # w is the output vector for the 2x2 multiplication
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # M1 -> C00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3 -> C01
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # M4 -> C00
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # M5 -> C00
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6 -> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]), # M7 -> C00
    ]
    
    # Wait, the w vectors in strassen_terms are not the full output.
    # The output C is a sum of these terms.
    # C = sum_t (u_t \otimes v_t \otimes w_t)
    # This is correct for the 2x2 case.
    
    # For 4x4, we have:
    # C = sum_{t=0}^6 sum_{s=0}^6 (U_{t,s} \otimes V_{t,s} \otimes W_{t,s})
    
    # Top-level patterns
    top_patterns = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),          # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),          # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}), # M7
    ]
    
    # Assembly of P_t into C
    # C = sum_t A_t P_t
    # A_t are 2x2 matrices of coefficients
    assembly = [
        np.array([[1, 0], [0, 1]]), # M1
        np.array([[0, 0], [1, -1]]), # M2
        np.array([[0, 1], [0, 1]]),  # M3
        np.array([[1, 0], [1, 0]]),  # M4
        np.array([[-1, 1], [0, 0]]), # M5
        np.array([[0, 0], [0, 1]]),  # M6
        np.array([[1, 0], [0, 0]]),  # M7
    ]
    
    col = 0
    for t_idx, (X_coeffs, Y_coeffs) in enumerate(top_patterns):
        A_t = assembly[t_idx]
        for u, v, w in strassen_terms:
            # u, v, w are length 4 vectors
            # w[k] is the coefficient for the k-th entry of P_t
            # P_t is a 2x2 matrix.
            # The k-th entry of P_t contributes to C with coefficient A_t[r, c]
            # where k = r*2 + c.
            
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # Map u to U_vec
            for k in range(4):
                if u[k] != 0:
                    for (i, j), c in X_coeffs.items():
                        if c != 0:
                            r = k // 2
                            c_loc = k % 2
                            g_row = i * 2 + r
                            g_col = j * 2 + c_loc
                            g_idx = g_row * 4 + g_col
                            U_vec[g_idx] += u[k] * c
            
            # Map v to V_vec
            for k in range(4):
                if v[k] != 0:
                    for (i, j), c in Y_coeffs.items():
                        if c != 0:
                            r = k // 2
                            c_loc = k % 2
                            g_row = i * 2 + r
                            g_col = j * 2 + c_loc
                            g_idx = g_row * 4 + g_col
                            V_vec[g_idx] += v[k] * c
            
            # Map w to W_vec
            for k in range(4):
                if w[k] != 0:
                    r = k // 2
                    c_loc = k % 2
                    # P_t[r, c] contributes to C with coeff A_t[r, c]
                    coeff = A_t[r, c_loc] * w[k]
                    if coeff != 0:
                        # C entry index is r, c_loc?
                        # No, C is 4x4.
                        # P_t is a 2x2 matrix of 2x2 blocks?
                        # No, P_t is a 2x2 matrix.
                        # The entries of P_t are scalars?
                        # No, in the recursive view, P_t is a 2x2 matrix over the ring of 2x2 matrices?
                        # No, P_t is a 2x2 matrix of scalars?
                        # Let's re-verify.
                        
                        # A is 4x4. Blocks A_ij are 2x2.
                        # X_t is a 2x2 matrix (sum of blocks).
                        # Y_t is a 2x2 matrix.
                        # P_t = X_t Y_t is a 2x2 matrix.
                        # The entries of P_t are scalars.
                        # C is a 4x4 matrix.
                        # C is assembled from P_t's.
                        # C = sum_t A_t P_t.
                        # A_t is a 2x2 matrix of coefficients.
                        # So C is a 2x2 matrix of 2x2 blocks?
                        # No, C is a 4x4 matrix.
                        # The assembly C = sum_t A_t P_t means:
                        # C_{block_i, block_j} = sum_t (A_t)_{i,j} P_t
                        # But P_t is a 2x2 matrix.
                        # So C_{block_i, block_j} is a 2x2 matrix?
                        # No, C is 4x4.
                        # The blocks of C are C_{ij} (2x2).
                        # C_{ij} = sum_t (A_t)_{i,j} P_t ?
                        # This would imply C_{ij} is a scalar if P_t is scalar.
                        # But P_t is a 2x2 matrix.
                        # So C_{ij} is a 2x2 matrix?
                        # No, C is 4x4.
                        # The "blocks" of C are 2x2 matrices.
                        # So C_{ij} (block) is a 2x2 matrix.
                        # This matches.
                        
                        # So for each entry k of P_t (which is a scalar),
                        # it contributes to the block C_{r, c_loc} (where r, c_loc are block indices).
                        # Specifically, to the entry (r, c_loc) of the block?
                        # No, P_t is a 2x2 matrix.
                        # P_t[k] is the (r, c_loc)-th entry of P_t.
                        # This entry is a scalar.
                        # It contributes to the (r, c_loc)-th block of C?
                        # No, the assembly A_t P_t is a 2x2 matrix of scalars.
                        # This 2x2 matrix is added to C?
                        # No, C is 4x4.
                        
                        # Let's trace indices carefully.
                        # A = [A00 A01; A10 A11]
                        # B = [B00 B01; B10 B11]
                        # C = [C00 C01; C10 C11]
                        # C00 = M1 + M4 - M5 + M7
                        # Each M_t is a 2x2 matrix.
                        # So C00 is a 2x2 matrix.
                        # This is correct.
                        
                        # So P_t[k] is the (r, c_loc)-th entry of the 2x2 matrix P_t.
                        # This entry is a scalar.
                        # It contributes to the (r, c_loc)-th entry of the block C_{r, c_loc}?
                        # No, P_t is the whole 2x2 matrix.
                        # The assembly is C_{block_i, block_j} = sum_t (A_t)_{i,j} P_t.
                        # So the entire 2x2 matrix P_t is multiplied by (A_t)_{i,j} and added to the block C_{i,j}.
                        
                        # So for a fixed t and s:
                        # We have a scalar product u_dot_v.
                        # This scalar is P_t[k] (the k-th entry of P_t).
                        # This scalar is added to the block C_{r, c_loc} with coefficient (A_t)_{r, c_loc} * w[k].
                        # Wait, w[k] is the coefficient in the Strassen decomposition of P_t.
                        # So P_t = sum_s (u_s \cdot v_s) w_s.
                        # So P_t[k] = sum_s (u_s \cdot v_s) w_s[k].
                        # And C_{r, c_loc} = sum_t (A_t)_{r, c_loc} P_t.
                        # So C_{r, c_loc}[entry] = sum_t sum_s (A_t)_{r, c_loc} w_s[k] (u_s \cdot v_s)
                        # where k is the entry index in P_t corresponding to the entry in C_{r, c_loc}?
                        # No, P_t is a 2x2 matrix.
                        # C_{r, c_loc} is a 2x2 matrix.
                        # C_{r, c_loc} = (A_t)_{r, c_loc} P_t.
                        # So the (a, b)-th entry of C_{r, c_loc} is (A_t)_{r, c_loc} P_t[a, b].
                        
                        # So for a fixed t, s, and entry k of P_t (k = a*2+b):
                        # The scalar (u_s \cdot v_s) w_s[k] is added to the (a, b)-th entry of the block C_{r, c_loc}
                        # with coefficient (A_t)_{r, c_loc}.
                        
                        # So the global index is:
                        # Block row r, Block col c_loc.
                        # Entry row a, Entry col b.
                        # Global row = r*2 + a
                        # Global col = c_loc*2 + b
                        # Global idx = (r*2+a)*4 + (c_loc*2+b)
                        
                        r = k // 2
                        c_loc = k % 2
                        coeff = A_t[r, c_loc] * w[k]
                        if coeff != 0:
                            # This scalar contributes to ALL entries of the block C_{r, c_loc}?
                            # No, P_t[k] is the (r, c_loc)-th entry of P_t.
                            # P_t is a 2x2 matrix.
                            # So P_t[k] is a scalar.
                            # It contributes to the (r, c_loc)-th block of C?
                            # No, C_{r, c_loc} is a 2x2 matrix.
                            # C_{r, c_loc} = (A_t)_{r, c_loc} P_t.
                            # So the (a, b)-th entry of C_{r, c_loc} is (A_t)_{r, c_loc} P_t[a, b].
                            # So P_t[k] (which is P_t[a, b]) contributes to the (a, b)-th entry of the block C_{r, c_loc}.
                            # Wait, r, c_loc are the block indices.
                            # a, b are the entry indices within the block.
                            # But k = a*2+b.
                            # So P_t[k] is P_t[a, b].
                            # It contributes to the (a, b)-th entry of the block C_{r, c_loc}.
                            # But r, c_loc are also a, b?
                            # No, r, c_loc are the block indices (0 or 1).
                            # a, b are the entry indices (0 or 1).
                            # So k = a*2+b.
                            # The block index is r, c_loc?
                            # No, the assembly is C_{block_i, block_j} = sum_t (A_t)_{i,j} P_t.
                            # So the block index is i, j.
                            # The entry index is a, b.
                            # So P_t[a, b] contributes to the (a, b)-th entry of the block C_{i, j}.
                            # And the coefficient is (A_t)_{i, j}.
                            
                            # So for a fixed t, s, and k (which is a*2+b):
                            # The scalar is (u_s \cdot v_s) w_s[k].
                            # It contributes to the (a, b)-th entry of the block C_{i, j}
                            # for all i, j?
                            # No, the coefficient is (A_t)_{i, j}.
                            # So we need to iterate over i, j.
                            
                            for i in range(2):
                                for j in range(2):
                                    coeff_ij = A_t[i, j] * w[k]
                                    if coeff_ij != 0:
                                        # Global index for entry (a, b) of block (i, j)
                                        g_row = i * 2 + a
                                        g_col = j * 2 + b
                                        g_idx = g_row * 4 + g_col
                                        W_vec[g_idx] += coeff_ij
            
            # Store in U, V, W
            for i in range(16):
                if U_vec[i] != 0:
                    U[i][col] = U_vec[i]
                if V_vec[i] != 0:
                    V[i][col] = V_vec[i]
                if W_vec[i] != 0:
                    W[i][col] = W_vec[i]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
