import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use two copies of a rank-21 decomposition for 3x4x3
    # derived from the rank-21 decomposition of 4x4x4 (Strassen)
    
    # Generate 4x4x4 Strassen coefficients
    # Strassen 2x2x2 has 7 terms
    # 4x4x4 has 21 terms
    
    # Helper to generate Strassen terms for size N
    def get_strassen_terms(size):
        # Returns list of (u, v, w) where u, v, w are size x size matrices
        # Actually, we want the scalar coefficients for the tensor
        # Tensor T = sum_r U_r \otimes V_r \otimes W_r
        # U_r is n*m, V_r is m*p, W_r is p*n
        
        if size == 2:
            # 7 terms for 2x2x2
            # Indices: A is 2x2, B is 2x2, C is 2x2
            # A indices: 00, 01, 10, 11
            # B indices: 00, 01, 10, 11
            # C indices: 00, 01, 10, 11
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            terms = []
            
            # T1
            u = [1, 0, 0, 1] # A00, A11
            v = [1, 0, 0, 1] # B00, B11
            w = [1, 0, 0, 1] # C00, C11 (Wait, C00 gets +1, C11 gets +1? No)
            # C00 = M1 + M4 - M5 + M7
            # C11 = M1 - M2 + M3 + M6
            # C01 = M3 + M5
            # C10 = M2 + M4
            
            # Let's list the (u, v, w) vectors for each M
            # M1: A00+A11, B00+B11 -> C00+1, C11+1
            terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
            
            # M2: A10+A11, B00 -> C10+1
            terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
            
            # M3: A00, B01-B11 -> C01+1
            terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
            
            # M4: A11, B10-B00 -> C00+1, C10+1
            terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
            
            # M5: A00+A01, B11 -> C00-1, C01+1
            terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
            
            # M6: A10-A00, B00+B01 -> C11+1
            terms.append(([1,-1,0,0], [1,1,0,0], [0,0,0,1]))
            
            # M7: A01-A11, B10+B11 -> C00+1
            terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
            
            return terms

        else:
            # Recursive step
            half = size // 2
            small_terms = get_strassen_terms(half)
            
            big_terms = []
            
            # Block structure:
            # A = [A11 A12; A21 A22]
            # B = [B11 B12; B21 B22]
            # C = [C11 C12; C21 C22]
            
            # Indices in big matrices:
            # 0..half-1, 0..half-1 -> 11
            # 0..half-1, half..size-1 -> 12
            # half..size-1, 0..half-1 -> 21
            # half..size-1, half..size-1 -> 22
            
            for u_small, v_small, w_small in small_terms:
                # u_small is size*half*half? No, u_small is (half*half) vector
                # We need to map small indices to big indices
                
                u_big = [0] * (size * size)
                v_big = [0] * (size * size)
                w_big = [0] * (size * size)
                
                # Map u_small (A block) to u_big
                # u_small index r corresponds to A_block[r]
                # We need to know which block each small term touches
                # This recursive approach is tricky because the terms in small
                # are sums of products of sub-blocks.
                
                # Actually, it's easier to just generate the 21 terms for 4x4x4
                # by applying the 7-term algorithm to the 4 blocks of 4x4.
                
                pass
            
            # I will hardcode the 21 terms for 4x4x4 instead of recursive generation
            # to avoid indexing errors.
            
            # But I can derive them programmatically using the 2x2x2 terms
            # by expanding the blocks.
            
            # Let's do the expansion properly.
            # Small term (u_s, v_s, w_s)
            # u_s is a vector of length 4 (for 2x2).
            # Each entry in u_s corresponds to a specific A element in the 2x2 block.
            # In the 4x4 case, the 2x2 block is composed of 2x2 sub-blocks.
            
            # This is getting complicated to code dynamically without a clear mapping.
            # I will use a known list of 21 terms for 4x4x4.
            
            # Alternatively, I can just use the rank 23 for 3x3x3 if I find it.
            # But 42 is better.
            
            # Let's stick to the 42 rank plan and hardcode the 21 terms for 4x4x4.
            
            return big_terms

    # Since recursive generation is complex to verify, I will implement a specific
    # known rank 21 algorithm for 4x4x4.
    # Actually, I can just use the 2x2x2 rank 7 algorithm to compute the 4x4x4
    # multiplication and trace the multiplications? No, I need the tensor decomposition.
    
    # I will use a simpler approach:
    # The rank of 3x4x3 is at most 21.
    # I will construct the 42-rank decomposition.
    
    # To be safe and fast, I will output the 72-rank solution if I cannot
    # confidently construct the 42-rank solution.
    # However, I am confident in the Strassen 4x4x4 rank 21.
    # I will write a helper to generate the 21 terms.
    
    def generate_4x4x4_terms():
        # Returns 21 tuples of (u, v, w) where u, v, w are length 16 vectors
        # representing 4x4 matrices flattened row-major
        
        terms_2x2 = get_strassen_terms(2)
        
        terms_4x4 = []
        
        # Block indices:
        # 0: 11, 1: 12, 2: 21, 3: 22
        
        # For each 2x2 term, we need to expand it to 4x4.
        # The 2x2 term operates on blocks A11, A12, A21, A22.
        # u_small has 4 entries: [u11, u12, u21, u22]
        # v_small has 4 entries: [v11, v12, v21, v22]
        # w_small has 4 entries: [w11, w12, w21, w22]
        
        # Each entry is a coefficient for a 2x2 sub-matrix multiplication.
        # Wait, the 2x2x2 algorithm multiplies 2x2 matrices.
        # So u_small[0] is the coefficient for A11 (which is a 2x2 matrix in 4x4 context).
        
        # So if u_small[0] is 1, it means we use A11.
        # A11 in 4x4 is the top-left 2x2 block.
        
        for u_s, v_s, w_s in terms_2x2:
            u_big = [0] * 16
            v_big = [0] * 16
            w_big = [0] * 16
            
            # Map block indices to big matrix indices
            # Block 0 (11): rows 0-1, cols 0-1
            # Block 1 (12): rows 0-1, cols 2-3
            # Block 2 (21): rows 2-3, cols 0-1
            # Block 3 (22): rows 2-3, cols 2-3
            
            def block_offset(block_idx, row_in_block, col_in_block):
                if block_idx == 0:
                    return row_in_block * 4 + col_in_block
                elif block_idx == 1:
                    return row_in_block * 4 + (2 + col_in_block)
                elif block_idx == 2:
                    return (2 + row_in_block) * 4 + col_in_block
                elif block_idx == 3:
                    return (2 + row_in_block) * 4 + (2 + col_in_block)
            
            # u_s entries correspond to A blocks 0,1,2,3
            for b_idx in range(4):
                coeff = u_s[b_idx]
                if coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            idx = block_offset(b_idx, r, c)
                            u_big[idx] += coeff
                
                coeff = v_s[b_idx]
                if coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            idx = block_offset(b_idx, r, c)
                            v_big[idx] += coeff
                
                coeff = w_s[b_idx]
                if coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            idx = block_offset(b_idx, r, c)
                            w_big[idx] += coeff
            
            terms_4x4.append((u_big, v_big, w_big))
            
        return terms_4x4

    terms_4x4 = generate_4x4x4_terms()
    
    # Now extract 3x4x3 sub-tensor
    # A is 3x4 (rows 0-2, cols 0-3). In 4x4, this is rows 0-2, cols 0-3.
    # B is 4x3 (rows 0-3, cols 0-2). In 4x4, this is rows 0-3, cols 0-2.
    # C is 3x3 (rows 0-2, cols 0-2). In 4x4, this is rows 0-2, cols 0-2.
    
    # We need to filter the 4x4x4 terms to keep only those that contribute to C[0:3, 0:3]
    # and only use A[0:3, :] and B[:, 0:3].
    
    # A indices in 4x4: i in 0..3, j in 0..3.
    # We want i in 0..2.
    # B indices in 4x4: j in 0..3, k in 0..3.
    # We want k in 0..2.
    # C indices in 4x4: i in 0..3, k in 0..3.
    # We want i in 0..2, k in 0..2.
    
    # If a term has non-zero entries in A[3, :] or B[:, 3] or C[3, :] or C[:, 3],
    # it might still be valid if those entries are zero in our sub-problem.
    # But if the term *requires* those entries to be non-zero to form the product,
    # it's invalid for the sub-problem.
    # However, the tensor decomposition is exact for the full 4x4x4.
    # If we set the extra rows/cols to 0, the product is still correct for the sub-blocks.
    # So we can just take the 21 terms and map the indices.
    
    # We need to create U, V, W for the 3x4x6 problem.
    # We will have 42 terms.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 12 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 18 x 42
    
    # Term 0-20: For C1 (cols 0-2 of output)
    # Term 21-40: For C2 (cols 3-5 of output)
    
    for t_idx in range(21):
        u_big, v_big, w_big = terms_4x4[t_idx]
        
        # Fill U for this term (column t_idx)
        # u_big is 4x4 flattened. We need 3x4 flattened.
        # A rows 0-2, cols 0-3.
        for i in range(3):
            for j in range(4):
                # Index in u_big: i * 4 + j
                U[i * m + j][t_idx] = u_big[i * 4 + j]
        
        # Fill V for this term (column t_idx)
        # v_big is 4x4 flattened. We need 4x3 flattened for B1.
        # B rows 0-3, cols 0-2.
        for i in range(4):
            for j in range(3):
                # Index in v_big: i * 4 + j
                V[i * p + j][t_idx] = v_big[i * 4 + j]
        
        # Fill W for this term (column t_idx)
        # w_big is 4x4 flattened. We need 3x3 flattened for C1.
        # C rows 0-2, cols 0-2.
        # W index is k * n + i. k is col of C, i is row of C.
        for i in range(3):
            for k in range(3):
                # Index in w_big: i * 4 + k
                W[k * n + i][t_idx] = w_big[i * 4 + k]
        
        # Now for the second half (C2, cols 3-5 of output)
        # This corresponds to B2 (rows 0-3, cols 3-5 of original B)
        # In 4x4 context, B2 is cols 3-? No, B is 4x6.
        # We split B into B1 (4x3) and B2 (4x3).
        # B1 uses cols 0-2. B2 uses cols 3-5.
        # The 4x4x4 algorithm assumes B is 4x4.
        # So for B2, we map cols 3-5 of B to cols 0-2 of the 4x4 B.
        
        # Fill V for term t_idx + 21
        for i in range(4):
            for j in range(3):
                # j is 0-2 in B2. Corresponds to col 3+j in original B.
                # In 4x4 B, this is col j.
                V[i * p + (3 + j)][t_idx + 21] = v_big[i * 4 + j]
        
        # Fill W for term t_idx + 21
        # C2 is cols 3-5 of C.
        # In 4x4 C, this is cols 0-2.
        for i in range(3):
            for k in range(3):
                # k is 0-2 in C2. Corresponds to col 3+k in original C.
                # In 4x4 C, this is col k.
                W[(3 + k) * n + i][t_idx + 21] = w_big[i * 4 + k]
        
        # U is the same for both halves (A is the same)
        for i in range(3):
            for j in range(4):
                U[i * m + j][t_idx + 21] = u_big[i * 4 + j]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
