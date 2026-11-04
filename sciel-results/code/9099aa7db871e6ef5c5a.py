import time

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the 4x4x4 blocks and schoolbook for the 4x4x1 block.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 9
    # We split p=9 into 4+4+1
    # Part 1: 4x4 * 4x4 (Strassen rank 49)
    # Part 2: 4x4 * 4x4 (Strassen rank 49)
    # Part 3: 4x4 * 4x1 (Schoolbook rank 16)
    
    # Helper to get Strassen 2x2x2 U,V,W vectors
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0,1,2,3 (row major: 00, 01, 10, 11)
    # Indices for B: 0,1,2,3
    # Indices for C: 0,1,2,3
    
    def get_strassen_2x2():
        # Returns list of 7 tuples (u, v, w) where u,v,w are lists of 4 elements
        # u corresponds to A, v to B, w to C
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to express each M_k as sum of rank 1 terms?
        # No, M_k IS a product of two 2x2 matrices.
        # We want to decompose the 2x2x2 tensor.
        # The 2x2x2 tensor rank is 7.
        # The 7 terms are the M_k products.
        # But M_k is a bilinear form.
        # We need to express the final C entries as linear combinations of M_k.
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        # The "rank-1" terms in the tensor decomposition of the 2x2x2 multiplication
        # are actually the 7 products M_k.
        # Each M_k corresponds to a specific linear combination of A's entries and B's entries.
        
        # Let's define the 7 U, V vectors for the 2x2 case.
        # u is a 4-dim vector (A entries), v is a 4-dim vector (B entries).
        # w is a 4-dim vector (C entries).
        
        # For M1:
        # u1 = [1, 0, 0, 1] (A00 + A11)
        # v1 = [1, 0, 0, 1] (B00 + B11)
        # The product M1 contributes to C.
        # How to map M1 to C?
        # M1 is a scalar in the "block" sense, but here it's the product of two vectors?
        # No, in the 2x2 case, the "scalars" are the entries.
        # M1 = (A00+A11)(B00+B11). This is a sum of 4 rank-1 terms if expanded?
        # No, the rank of the 2x2x2 tensor is 7.
        # The 7 terms are the M_k.
        # Each M_k is a rank-1 term in the decomposition of the tensor?
        # Yes, if we view the inputs as the linear combinations.
        
        # Let's verify.
        # Tensor T = sum_{i,j,k} A_ik B_kj C_ij
        # We write T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
        # u_r is a vector in R^4 (coefficients for A)
        # v_r is a vector in R^4 (coefficients for B)
        # w_r is a vector in R^4 (coefficients for C)
        
        # For M1:
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        # The product is (A00+A11)(B00+B11).
        # This product appears in C with specific coefficients.
        # C00 += M1
        # C11 += M1
        # So w1 = [1, 0, 0, 1]
        
        # Let's check M2:
        # M2 = (A10+A11)B00
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # C10 += M2
        # w2 = [0, 0, 1, 0]
        
        # M3:
        # M3 = A00(B01-B11)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # C01 += M3
        # C11 -= M3
        # w3 = [0, 1, 0, -1]
        
        # M4:
        # M4 = A11(B10-B00)
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1]  # B10 - B00 -> [0,0,1,0] - [1,0,0,0] = [-1,0,1,0]?
        # Wait, B indices: 00, 01, 10, 11.
        # B10 is index 2. B00 is index 0.
        # v4 = [-1, 0, 1, 0]
        # C00 -= M4
        # C10 += M4
        # w4 = [-1, 0, 1, 0]
        
        # M5:
        # M5 = (A00+A01)B11
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # C01 += M5
        # C11 += M5
        # w5 = [0, 1, 0, 1]
        
        # M6:
        # M6 = (A10-A00)(B00+B01)
        # u6 = [-1, 0, 1, 0]
        # v6 = [1, 1, 0, 0]
        # C00 += M6
        # w6 = [1, 0, 0, 0]
        
        # M7:
        # M7 = (A01-A11)(B10+B11)
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # C00 -= M7
        # C11 += M7
        # w7 = [-1, 0, 0, 1]
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])
        ]

    # Now build 4x4x4 decomposition
    # We have 7 top-level terms.
    # For each term k, we have U_k (4x4 matrix) and V_k (4x4 matrix).
    # We decompose U_k * V_k using the 2x2 Strassen logic.
    # This means we treat U_k and V_k as 2x2 block matrices.
    
    # The 2x2 Strassen logic gives 7 sub-terms.
    # Each sub-term corresponds to a specific linear combination of the blocks of U_k and V_k.
    
    # Let's pre-compute the 49 terms.
    
    strassen_2x2 = get_strassen_2x2()
    
    # Top level Strassen definitions for 4x4
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11
    
    # S1 = A00 + A11
    # T1 = B00 + B11
    # S2 = A10 + A11
    # T2 = B00
    # S3 = A00
    # T3 = B01 - B11
    # S4 = A11
    # T4 = B10 - B00
    # S5 = A00 + A01
    # T5 = B11
    # S6 = A10 - A00
    # T6 = B00 + B01
    # S7 = A01 - A11
    # T7 = B10 + B11
    
    # We represent U_k and V_k as 4x4 matrices (lists of lists or flat lists)
    # But we need to feed them into the 2x2 Strassen logic.
    # The 2x2 Strassen logic expects 4 scalars.
    # Here the "scalars" are 2x2 blocks.
    
    # So, for each top-level term k:
    # We have 4 "block-scalars" for U_k: [U_k_00, U_k_01, U_k_10, U_k_11]
    # And 4 for V_k.
    
    # We iterate over the 7 sub-terms of the 2x2 Strassen.
    # For each sub-term j:
    # We have u_j (4 coeffs), v_j (4 coeffs), w_j (4 coeffs).
    # u_j defines a linear combination of the 4 blocks of U_k.
    # This results in a 4x4 matrix (block vector).
    # Similarly for v_j and w_j.
    
    # Finally, we flatten these 4x4 matrices to 16-dim vectors.
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    # Define the 7 top-level block combinations
    # Each is a list of 4 coefficients for the 4 blocks (00, 01, 10, 11)
    top_u_blocks = [
        [1, 0, 0, 1], # S1: A00 + A11
        [0, 0, 1, 1], # S2: A10 + A11
        [1, 0, 0, 0], # S3: A00
        [0, 0, 0, 1], # S4: A11
        [1, 1, 0, 0], # S5: A00 + A01
        [-1, 0, 1, 0],# S6: A10 - A00
        [0, 1, 0, -1] # S7: A01 - A11
    ]
    
    top_v_blocks = [
        [1, 0, 0, 1], # T1: B00 + B11
        [1, 0, 0, 0], # T2: B00
        [0, 1, 0, -1],# T3: B01 - B11
        [-1, 0, 1, 0],# T4: B10 - B00
        [0, 0, 0, 1], # T5: B11
        [1, 1, 0, 0], # T6: B00 + B01
        [0, 0, 1, 1]  # T7: B10 + B11
    ]
    
    # For each of the 7 top-level terms
    for k in range(7):
        # For each of the 7 sub-terms (from 2x2 Strassen)
        for j in range(7):
            u_sub, v_sub, w_sub = strassen_2x2[j]
            
            # Construct the 4x4 U vector
            # u_sub has 4 coefficients c0, c1, c2, c3
            # U_k has 4 blocks B00, B01, B10, B11
            # The resulting 4x4 matrix is:
            # [ c0*U_k_00 + c1*U_k_01? No.
            # The blocks of U_k are linear combinations of A's blocks.
            # Let's be precise.
            
            # U_k is a 4x4 matrix.
            # Its (0,0) block is sum_{b=0}^3 top_u_blocks[k][b] * A_block_b
            # Wait, top_u_blocks[k] gives coefficients for A's blocks.
            # So U_k_block_00 = top_u_blocks[k][0]*A00 + top_u_blocks[k][1]*A01 + ...
            
            # The 2x2 Strassen sub-term j combines the 4 blocks of U_k.
            # u_sub[0] is coeff for U_k_00.
            # So the final U vector is:
            # U_vec_block_00 = u_sub[0] * (top_u_blocks[k][0]*A00 + top_u_blocks[k][1]*A01 + top_u_blocks[k][2]*A10 + top_u_blocks[k][3]*A11)
            # This is getting complicated to track indices.
            
            # Easier:
            # U_k is a 4x4 matrix formed by A.
            # Let's just build the 4x4 matrix for U_k explicitly for each k.
            
    # Let's restart the construction logic more simply.
    
    # Build U, V, W for 4x4x4
    U_4x4_terms = []
    V_4x4_terms = []
    W_4x4_terms = []
    
    # A indices: 0..15
    # B indices: 0..15
    # C indices: 0..15
    
    # For each of the 49 terms:
    for k in range(7):
        for j in range(7):
            u_sub, v_sub, w_sub = strassen_2x2[j]
            
            # u_sub, v_sub, w_sub are 4-dim vectors.
            # They define linear combinations of the 4 blocks of the "virtual" 2x2 matrices.
            
            # The "virtual" U matrix (4x4) is formed by A.
            # U_virtual_block_00 = sum_{b=0}^3 top_u_blocks[k][b] * A_block_b
            # ...
            
            # The final U term (4x4) is:
            # U_term_block_00 = u_sub[0] * U_virtual_block_00 + u_sub[1] * U_virtual_block_01 + ...
            
            # This is equivalent to:
            # U_term = sum_{i=0}^3 u_sub[i] * U_virtual_block_i
            
            # And U_virtual_block_i is a linear combination of A's blocks.
            
            # So U_term is a linear combination of A's blocks.
            
            # Let's compute the coefficients for A's 16 entries.
            
            # A blocks:
            # A00: A[0:2, 0:2]
            # A01: A[0:2, 2:4]
            # A10: A[2:4, 0:2]
            # A11: A[2:4, 2:4]
            
            # u_final is a 16-dim vector.
            u_final = [0.0] * 16
            v_final = [0.0] * 16
            w_final = [0.0] * 16
            
            # For each block position (r, c) in the 4x4 matrix (r in 0..1, c in 0..1 for blocks)
            # The block is at rows 2r..2r+1, cols 2c..2c+1.
            
            # The coefficient for A_block_b in U_virtual_block_{r,c} is top_u_blocks[k][b]?
            # No.
            # top_u_blocks[k] is a vector of 4 coeffs for the 4 blocks of A.
            # But U_virtual is a 2x2 block matrix.
            # Which block of U_virtual corresponds to which block of A?
            # In Strassen, S1 = A00 + A11.
            # S1 is a 2x2 block matrix where (1,1) is A00 and (2,2) is A11.
            # So U_virtual_block_00 = A00.
            # U_virtual_block_11 = A11.
            # U_virtual_block_01 = 0.
            # U_virtual_block_10 = 0.
            
            # So we need a mapping from top-level term k to the 4 blocks of U_virtual.
            
            # Let's define the block structure for each k.
            # k=0: S1 = A00 + A11. Blocks: [A00, 0, 0, A11]
            # k=1: S2 = A10 + A11. Blocks: [0, 0, A10, A11]
            # k=2: S3 = A00. Blocks: [A00, 0, 0, 0]
            # k=3: S4 = A11. Blocks: [0, 0, 0, A11]
            # k=4: S5 = A00 + A01. Blocks: [A00, A01, 0, 0]
            # k=5: S6 = A10 - A00. Blocks: [-A00, 0, A10, 0]
            # k=6: S7 = A01 - A11. Blocks: [0, A01, 0, -A11]
            
            # Let's hardcode these block assignments.
            # block_assignment[k] is a 4x4 matrix of indices into A's blocks?
            # Or just a 4x4 matrix of coefficients for A's 4 blocks.
            
            a_block_coeffs = [
                [[1, 0], [0, 1]], # k=0: A00 at (0,0), A11 at (1,1)
                [[0, 0], [1, 1]], # k=1: A10 at (1,0), A11 at (1,1)
                [[1, 0], [0, 0]], # k=2: A00 at (0,0)
                [[0, 0], [0, 1]], # k=3: A11 at (1,1)
                [[1, 1], [0, 0]], # k=4: A00 at (0,0), A01 at (0,1)
                [[-1, 0], [1, 0]],# k=5: -A00 at (0,0), A10 at (1,0)
                [[0, 1], [0, -1]] # k=6: A01 at (0,1), -A11 at (1,1)
            ]
            
            b_block_coeffs = [
                [[1, 0], [0, 1]], # k=0: B00 at (0,0), B11 at (1,1)
                [[1, 0], [0, 0]], # k=1: B00 at (0,0)
                [[0, 1], [0, -1]],# k=2: B01 at (0,1), -B11 at (1,1)
                [[-1, 0], [1, 0]],# k=3: -B00 at (0,0), B10 at (1,0)
                [[0, 0], [0, 1]], # k=4: B11 at (1,1)
                [[1, 1], [0, 0]], # k=5: B00 at (0,0), B01 at (0,1)
                [[0, 0], [1, 1]]  # k=6: B10 at (1,0), B11 at (1,1)
            ]
            
            # Now compute u_final
            # u_final is a 16-dim vector.
            # For each block position (br, bc) in 0..1:
            #   The coefficient for A_block_b in U_virtual_block_{br,bc} is a_block_coeffs[k][br][bc]
            #   The sub-term j uses u_sub[br*2+bc] as the coefficient for U_virtual_block_{br,bc}
            #   So the total coefficient for A_block_b in the final U is:
            #   sum_{br,bc} u_sub[br*2+bc] * a_block_coeffs[k][br][bc]
            
            # Wait, u_sub is a 4-dim vector corresponding to the 4 blocks of the 2x2 matrix.
            # Index 0: block (0,0)
            # Index 1: block (0,1)
            # Index 2: block (1,0)
            # Index 3: block (1,1)
            
            # So for each A_block_b (0..3):
            #   coeff = sum_{br=0..1, bc=0..1} u_sub[br*2+bc] * a_block_coeffs[k][br][bc]
            #   This coeff applies to all 4 entries of A_block_b in u_final.
            
            a_block_indices = [
                [0, 1, 4, 5],   # A00
                [2, 3, 6, 7],   # A01
                [8, 9, 12, 13], # A10
                [10, 11, 14, 15]# A11
            ]
            
            for b in range(4):
                coeff = 0.0
                for br in range(2):
                    for bc in range(2):
                        coeff += u_sub[br*2+bc] * a_block_coeffs[k][br][bc]
                if coeff != 0:
                    for idx in a_block_indices[b]:
                        u_final[idx] += coeff
            
            # Similarly for v_final
            for b in range(4):
                coeff = 0.0
                for br in range(2):
                    for bc in range(2):
                        coeff += v_sub[br*2+bc] * b_block_coeffs[k][br][bc]
                if coeff != 0:
                    for idx in a_block_indices[b]: # Same indices for B
                        v_final[idx] += coeff
            
            # For w_final, the structure is the same as C.
            # C blocks are C00, C01, C10, C11.
            # The w_sub vector gives coefficients for the 4 blocks of the result.
            # The result blocks are linear combinations of C's blocks.
            # In Strassen, the result of the 2x2 product is the 2x2 matrix C.
            # So w_final is just w_sub applied to the C blocks.
            
            for b in range(4):
                coeff = w_sub[b]
                if coeff != 0:
                    for idx in a_block_indices[b]:
                        w_final[idx] += coeff
            
            U_4x4_terms.append(u_final)
            V_4x4_terms.append(v_final)
            W_4x4_terms.append(w_final)
            
    # Now we have 49 terms for 4x4x4.
    # We need two such sets (for B1 and B2).
    # And 16 terms for 4x4x1.
    
    # Total U, V, W matrices
    # U: (16, 114)
    # V: (36, 114)  # B is 4x9, so 36 entries
    # W: (36, 114)  # C is 4x9, so 36 entries
    
    U = [[0.0] * 114 for _ in range(16)]
    V = [[0.0] * 114 for _ in range(36)]
    W = [[0.0] * 114 for _ in range(36)]
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (B columns 0..3)
    for t in range(49):
        for i in range(16):
            U[i][col] = U_4x4_terms[t][i]
        for i in range(16):
            V[i][col] = V_4x4_terms[t][i]
        for i in range(16):
            W[i][col] = W_4x4_terms[t][i]
        col += 1
        
    # Part 2: 4x4 * 4x4 (B columns 4..7)
    # B is 4x9.
    # B1 is cols 0-3. B2 is cols 4-7. B3 is col 8.
    # For Part 2, V indices are shifted by 16?
    # B is 4x9. Row-major.
    # B[0:4, 0:4] is indices 0..15.
    # B[0:4, 4:8] is indices 16..31.
    # B[0:4, 8:9] is indices 32..35.
    
    for t in range(49):
        for i in range(16):
            U[i][col] = U_4x4_terms[t][i]
        for i in range(16):
            V[i+16][col] = V_4x4_terms[t][i]
        for i in range(16):
            W[i+16][col] = W_4x4_terms[t][i]
        col += 1
        
    # Part 3: 4x4 * 4x1 (B column 8)
    # Schoolbook rank 16.
    # C_i = sum_k A_ik B_k8
    # Terms: for each i, k: A_ik * B_k8 * e_i
    
    for i in range(4):
        for k in range(4):
            u_idx = i * 4 + k
            v_idx = k * 9 + 8 # B is 4x9. Row k, Col 8. Index k*9+8
            w_idx = i * 9 + 8 # C is 4x9. Row i, Col 8. Index i*9+8
            
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}

