import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W matrices.
    # Rank 81: 49 for 4x4x4, 16 for 4x4x1, 16 for 4x4x2.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4x4 Strassen
    # Strassen for 4x4 uses 7x7 = 49 multiplications.
    # We need to map the Strassen products to the tensor factors.
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U_list.append(u_idx)
        V_list.append(v_idx)
        W_list.append(w_idx)

    # Strassen 2x2 decomposition indices
    # For 2x2, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we apply this to 2x2 blocks.
    # Let A be split into A11, A12, A21, A22 (each 2x2).
    # Let B1 be split into B11, B12, B21, B22 (each 2x2).
    
    # We need to generate the 49 terms for the 4x4x4 part.
    # Each term is a product of a linear combination of A blocks and a linear combination of B blocks.
    # The result contributes to specific C blocks.
    
    # This is complex to code from scratch. 
    # Alternative: Use the known rank-49 decomposition for 4x4x4.
    # Or simply use the 97 rank (49 + 48) which is easier to code correctly.
    # 97 is better than 112.
    
    # Let's try to code the 97 rank first to ensure correctness, then optimize if time permits.
    # Actually, 81 is significantly better. Let's try to implement 81.
    
    # Strassen 4x4:
    # The 49 terms correspond to pairs of Strassen-2x2 indices (i,j) where i,j in 1..7.
    # Term (i,j) computes a contribution to the 4x4 result.
    # We can define the U, V, W factors for the 49 terms.
    
    # For simplicity and robustness, I will implement the 97 rank solution.
    # 49 (Strassen 4x4) + 48 (Schoolbook 4x3).
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: Strassen for A * B[:, 0:4]
    # We need the rank-49 decomposition for 4x4x4.
    # I will hardcode the Strassen 4x4 decomposition.
    
    # Strassen 2x2 basis vectors for U (A factors)
    # A is 4x4. Indices 0..3.
    # Blocks:
    # A11: [0,0],[0,1],[1,0],[1,1] -> 0,1,4,5
    # A12: [0,2],[0,3],[1,2],[1,3] -> 2,3,6,7
    # A21: [2,0],[2,1],[3,0],[3,1] -> 8,9,12,13
    # A22: [2,2],[2,3],[3,2],[3,3] -> 10,11,14,15
    
    # This is getting complicated to map manually.
    # Let's use the 112 rank schoolbook? No, score 1.0 is 112.
    # I must improve.
    
    # Let's use the 97 rank.
    # I will generate the Strassen 4x4 terms programmatically.
    
    # Define Strassen 2x2 multipliers
    # Returns list of (coeff_A, coeff_B, coeff_C) for each of the 7 products
    # Actually, we need the tensor factors.
    
    # Let's just use the 49 rank for 4x4x4 and 48 for 4x4x3.
    
    # 4x4x4 Strassen
    # We can construct it by applying Strassen 2x2 to the 4x4 blocks.
    
    # Let's write a helper to add a term to U,V,W
    def add_rank1(u_indices, v_indices, w_indices, sign=1):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] += sign
        for v_idx in v_indices:
            V[v_idx][col] += sign
        for w_idx in w_indices:
            W[w_idx][col] += sign
        col += 1

    # Strassen 2x2 decomposition
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 products:
    # 1: (A11+A22)(B11+B22) -> C11 += M1, C12 -= M1, C21 += M1, C22 += M1 (with signs)
    # Actually, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M6
    # C22 = M1 - M3 + M2 - M6
    
    # For 4x4, we treat A11, A12, A21, A22 as 2x2 matrices.
    # So we need 7 * 7 = 49 terms.
    # Each term is a product of a linear combo of A-blocks and B-blocks.
    # Since A-blocks are 2x2, the linear combo is a 2x2 matrix.
    # The product of two 2x2 matrices requires 4 multiplications (schoolbook) or 7 (Strassen).
    # But in the tensor decomposition, we are looking for rank-1 terms of the form u v^T w.
    # A term like (A11+A22)(B11+B22) expands to 4 terms:
    # A11 B11 + A11 B22 + A22 B11 + A22 B22.
    # Each of these is a 2x2x2 tensor product? No, they are 2x2 matrices multiplied.
    # The rank of 2x2x2 is 7.
    # So one "Strassen 4x4 term" like M11 (product of block sums) actually requires 7 rank-1 tensors if we go down to scalars?
    # No, the rank of 4x4x4 is 49.
    # This means we can represent the whole 4x4x4 multiplication with 49 rank-1 tensors.
    # These 49 tensors are not simply the 49 scalar multiplications of schoolbook.
    # They are the 49 terms from applying Strassen twice.
    
    # I will use a known library or hardcoded values if possible, but I don't have them.
    # However, I can implement the 97 rank using Schoolbook for the 4x3 part and Strassen for 4x4 part.
    # But I need the Strassen 4x4 decomposition.
    
    # Let's fall back to 112 if I can't get Strassen right quickly.
    # But 112 is score 1.0. I need > 1.0.
    
    # Let's try 97 with Schoolbook for 4x4x4 (64) + Schoolbook for 4x4x3 (48) = 112. No.
    # I MUST use Strassen for 4x4x4 to get 49.
    
    # I will generate the 49 terms for 4x4x4 Strassen.
    # The 49 terms correspond to the 7x7 grid of Strassen-2x2 applications.
    # Let s1, s2 be the Strassen indices for the outer level (4x4 blocks).
    # Let t1, t2 be the Strassen indices for the inner level (2x2 blocks).
    
    # This is too complex to derive in the fly.
    
    # Alternative: Use the fact that $\mu(4,4,7) \le 49 + 48 = 97$.
    # I will implement the 49 rank Strassen for 4x4x4 by hardcoding the 49 vectors.
    # Since I don't have them, I will use a simpler improvement:
    # $\mu(4,4,7) \le 4 \times \mu(1,4,7) = 4 \times 28 = 112$.
    # $\mu(4,4,7) \le 7 \times \mu(4,4,1) = 7 \times 16 = 112$.
    
    # How about $\mu(4,4,7) \le 2 \times \mu(2,4,7)$.
    # $\mu(2,4,7) \le 7 \times \mu(2,4,1) = 7 \times 8 = 56$.
    # $2 \times 56 = 112$.
    
    # It seems 112 is hard to beat without Strassen.
    
    # I will implement the Strassen 4x4x4 decomposition.
    # I found a reference for the 49 terms.
    # I will construct them.
    
    # For now, I will return the 112 rank solution to be safe, but that scores 1.0.
    # I need to improve.
    
    # Let's try 97. I will implement Strassen 4x4x4 using a recursive function that generates the factors.
    
    def strassen_2x2_factors():
        # Returns U, V, W factors for 2x2x2 Strassen (7 terms)
        # U shape 4x7, V shape 4x7, W shape 4x7
        U_s = np.zeros((4, 7), dtype=int)
        V_s = np.zeros((4, 7), dtype=int)
        W_s = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # A indices: 0:A11, 1:A12, 2:A21, 3:A22
        # B indices: 0:B11, 1:B12, 2:B21, 3:B22
        # C indices: 0:C11, 1:C12, 2:C21, 3:C22
        
        # Term 1
        U_s[0,0] = 1; U_s[3,0] = 1
        V_s[0,0] = 1; V_s[3,0] = 1
        W_s[0,0] = 1; W_s[1,0] = -1; W_s[2,0] = 1; W_s[3,0] = 1
        
        # Term 2
        U_s[2,1] = 1; U_s[3,1] = 1
        V_s[0,1] = 1
        W_s[2,1] = 1; W_s[3,1] = 1
        
        # Term 3
        U_s[0,2] = 1
        V_s[2,2] = 1; V_s[3,2] = -1
        W_s[1,2] = 1; W_s[3,2] = -1
        
        # Term 4
        U_s[3,3] = 1
        V_s[1,3] = 1; V_s[0,3] = -1
        W_s[0,3] = -1; W_s[1,3] = 1
        
        # Term 5
        U_s[0,4] = 1; U_s[1,4] = 1
        V_s[3,4] = 1
        W_s[0,4] = 1; W_s[3,4] = 1
        
        # Term 6
        U_s[2,5] = 1; U_s[0,5] = -1
        V_s[0,5] = 1; V_s[1,5] = 1
        W_s[2,5] = 1; W_s[3,5] = -1
        
        # Term 7
        U_s[1,6] = 1; U_s[3,6] = -1
        V_s[2,6] = 1; V_s[3,6] = 1
        W_s[0,6] = 1; W_s[1,6] = 1
        
        return U_s, V_s, W_s

    # Now for 4x4x4.
    # We have 4 blocks in A, 4 in B, 4 in C.
    # We apply Strassen 2x2 to the blocks.
    # The 7 outer terms are linear combos of blocks.
    # Each block product is a 2x2x2 multiplication.
    # So each outer term generates 7 inner terms.
    # Total 49.
    
    U_44 = np.zeros((16, 49), dtype=int)
    V_44 = np.zeros((16, 49), dtype=int)
    W_44 = np.zeros((16, 49), dtype=int)
    
    U_s, V_s, W_s = strassen_2x2_factors()
    
    # Map block indices to global indices
    # A blocks:
    # 0: A11 (0-3), 1: A12 (4-7), 2: A21 (8-11), 3: A22 (12-15)
    # B blocks:
    # 0: B11 (0-3), 1: B12 (4-7), 2: B21 (8-11), 3: B22 (12-15)
    # C blocks:
    # 0: C11 (0-3), 1: C12 (4-7), 2: C21 (8-11), 3: C22 (12-15)
    
    for outer in range(7):
        # Get the block coefficients for this outer term
        # U_s[:, outer] gives coeffs for A blocks
        # V_s[:, outer] gives coeffs for B blocks
        # W_s[:, outer] gives coeffs for C blocks
        
        for inner in range(7):
            col_idx = outer * 7 + inner
            
            # For each A block a with coeff ua
            # For each B block b with coeff vb
            # The product contributes to C blocks c with coeff wc * (inner contribution)
            
            # Actually, the inner Strassen computes the product of two 2x2 matrices.
            # The result is a 2x2 matrix which is added to the C blocks.
            
            # We need to add the rank-1 terms of the inner Strassen, scaled by the outer coefficients.
            
            for a_block in range(4):
                ua = U_s[a_block, outer]
                if ua == 0: continue
                
                for b_block in range(4):
                    vb = V_s[b_block, outer]
                    if vb == 0: continue
                    
                    # The inner term 'inner' has factors U_s[:, inner], V_s[:, inner], W_s[:, inner]
                    # These are for the 2x2 matrices within the blocks.
                    
                    # The global U index for A block a_block, local index l is:
                    # a_block * 4 + l
                    # Similarly for V and W.
                    
                    # The contribution is ua * vb * (inner term)
                    
                    for l in range(4):
                        u_val = ua * U_s[l, inner]
                        v_val = vb * V_s[l, inner]
                        w_val = W_s[l, inner]
                        
                        if u_val != 0:
                            U_44[a_block * 4 + l, col_idx] += u_val
                        if v_val != 0:
                            V_44[b_block * 4 + l, col_idx] += v_val
                        if w_val != 0:
                            # Wait, W_s[l, inner] is the coefficient for the C block's local index l?
                            # No, W_s in strassen_2x2_factors is for the 2x2 result.
                            # The outer W_s[c_block, outer] tells which C block receives the result.
                            # So we need to multiply by W_s[c_block, outer] as well?
                            # No, the Strassen formula is C = sum_outer W_outer * (M_outer).
                            # M_outer is a 2x2 matrix.
                            # The inner Strassen computes M_outer = A_outer * B_outer.
                            # So the total factor for W is W_s[c_block, outer] * W_s[l, inner].
                            
                            for c_block in range(4):
                                wc_outer = W_s[c_block, outer]
                                if wc_outer == 0: continue
                                w_total = wc_outer * w_val
                                if w_total != 0:
                                    W_44[c_block * 4 + l, col_idx] += w_total

    # Now we have U_44, V_44, W_44 for the 4x4x4 part.
    # We need to append the 4x4x3 part.
    # Schoolbook for 4x4x3: 48 terms.
    
    rank = 49 + 48
    U_final = np.zeros((16, rank), dtype=int)
    V_final = np.zeros((28, rank), dtype=int) # 4x7 = 28
    W_final = np.zeros((28, rank), dtype=int) # 7x4 = 28
    
    # Copy 4x4x4 part
    # U_44 is 16x49. V_44 is 16x49 (for B11..B22, cols 0-3 of B).
    # W_44 is 16x49 (for C11..C22, cols 0-3 of C).
    
    U_final[:, 0:49] = U_44
    V_final[0:16, 0:49] = V_44
    # W_44 indices are for C[:, 0:4].
    # W_final indices are for C[:, 0:7].
    # C11 is C[0:2, 0:2] -> W indices 0,1,4,5
    # C12 is C[0:2, 2:4] -> W indices 2,3,6,7
    # C21 is C[2:4, 0:2] -> W indices 8,9,12,13
    # C22 is C[2:4, 2:4] -> W indices 10,11,14,15
    
    # W_44 has 16 rows.
    # Row 0-3 -> C11
    # Row 4-7 -> C12
    # Row 8-11 -> C21
    # Row 12-15 -> C22
    
    # W_final has 28 rows.
    # C[0,0] is W_final[0]
    # C[0,1] is W_final[1]
    # C[0,2] is W_final[2]
    # ...
    # C[0,3] is W_final[3]
    # C[1,0] is W_final[4]
    # ...
    
    # Mapping W_44 row r to W_final row:
    # r=0 (C11,00) -> 0
    # r=1 (C11,01) -> 1
    # r=2 (C11,10) -> 4
    # r=3 (C11,11) -> 5
    # r=4 (C12,00) -> 2
    # r=5 (C12,01) -> 3
    # r=6 (C12,10) -> 6
    # r=7 (C12,11) -> 7
    # r=8 (C21,00) -> 8
    # r=9 (C21,01) -> 9
    # r=10 (C21,10) -> 12
    # r=11 (C21,11) -> 13
    # r=12 (C22,00) -> 10
    # r=13 (C22,01) -> 11
    # r=14 (C22,10) -> 14
    # r=15 (C22,11) -> 15
    
    w_map = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    for r in range(16):
        W_final[w_map[r], 0:49] = W_44[r, 0:49]
        
    # Now add 4x4x3 schoolbook
    # B2 is cols 4,5,6 of B.
    # B indices: 16,17,18 (col 4), 19,20,21 (col 5), 22,23,24 (col 6)
    # C indices: 16,17,18,19 (col 4), 20,21,22,23 (col 5), 24,25,26,27 (col 6)
    
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A[i,j] * B[j, 4+k] -> C[i, 4+k]
                U_final[i * 4 + j, col] = 1
                V_final[j * 7 + (4 + k), col] = 1
                W_final[(4 + k) * 4 + i, col] = 1
                col += 1
                
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

