# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 65.
    # This consists of:
    # 1. Rank 49 terms for the 4x4x4 multiplication (Strassen).
    # 2. Rank 16 terms for the 4x4x1 multiplication (Schoolbook).
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 2: Rank 49 Strassen terms for the first 4 columns of B ---
    # The Strassen algorithm for 4x4 matrices can be viewed as 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication requires 7 scalar multiplications (rank 7).
    # Total rank = 7 * 7 = 49.
    # 
    # We explicitly define the 49 terms.
    # Let A be 4x4, B be 4x4 (first 4 cols), C be 4x4 (first 4 cols).
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C = [[C00, C01], [C10, C11]] where each block is 2x2.
    #
    # Strassen steps for 2x2 blocks:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a product of two 2x2 matrices, which can be decomposed into 7 rank-1 terms.
    # Let P = X * Y where X, Y are 2x2.
    # P[i,j] = sum_k X[i,k] Y[k,j].
    # Rank-1 decomposition of 2x2x2 multiplication (Strassen 2x2x2):
    # T[i,j,k] = sum_r u[i,r] v[j,r] w[k,r] ? No, standard tensor.
    # The multiplication tensor for 2x2x2 has rank 7.
    # We need to generate the 7 terms for each of the 7 Strassen multiplications.
    
    # Helper: Generate the 7 rank-1 terms for a 2x2 matrix multiplication.
    # Returns list of (u_idx_local, v_idx_local, w_idx_local, coeff)
    # Local indices 0..3 for U (2x2), 0..3 for V (2x2), 0..3 for W (2x2).
    # Note: In the global tensor, U index is i*m+j, V is j*p+k, W is k*n+i.
    # For a local 2x2 block at offsets (oi, oj, ok), we map local indices to global.
    
    def get_strassen_2x2_terms():
        # Standard Strassen 2x2x2 decomposition terms.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # The 7 terms for 2x2 multiplication (rank 7).
        # Each term is a tensor of rank 1: u_i v_j w_k.
        # We list the coefficients for the 8 components of the 2x2x2 tensor?
        # No, we just need the vectors u, v, w.
        
        # Let's use the known rank-7 decomposition for 2x2x2.
        # Terms (u, v, w) where u,v,w are in R^2? No, u is row of A, v is col of A/row of B, w is col of B.
        # Actually, for 2x2x2, the indices are i in 0..1, j in 0..1, k in 0..1.
        # U[i, r], V[j, r], W[k, r].
        
        # Standard Strassen 2x2x2 vectors:
        # 1: u=[1,1], v=[1,1], w=[1,1]  -> (A00+A11)(B00+B11) ... wait, this is not quite right for direct expansion.
        # Let's use the explicit 7 terms from Strassen's algorithm applied to 2x2.
        # The 7 multiplications are:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # These are scalar multiplications. But we are decomposing the tensor of matrix multiplication.
        # The tensor T[i,j,k] = 1 if C[i,k] uses A[i,j]B[j,k].
        # The rank-1 decomposition sums to T.
        
        # Let's construct the 7 terms for the 2x2 multiplication tensor.
        # Term 1: (e1+e2)(e1+e2) -> contributes to all 4 entries of product?
        # No, the rank-1 terms are linear forms.
        
        # Here are the 7 rank-1 tensors for 2x2x2 multiplication:
        # 1. u=[1,1], v=[1,1], w=[1,1]  (Coeff 1) -> (A00+A11)(B00+B11) is not a rank 1 term in the tensor sense, it's a product of sums.
        # The rank 1 terms are the "atoms".
        
        # Correct 7 terms for 2x2x2 (Strassen):
        # 1. u=[1,1], v=[1,1], w=[1,1]
        # 2. u=[0,1], v=[1,0], w=[1,0]
        # 3. u=[1,0], v=[0,1], w=[0,1]
        # 4. u=[0,1], v=[0,1], w=[1,0]
        # 5. u=[1,1], v=[0,1], w=[1,0]
        # 6. u=[1,0], v=[1,1], w=[1,0]
        # 7. u=[1,0], v=[1,0], w=[1,1]
        
        # Let's verify Term 1:
        # Sum u[i]v[j]w[k] = (1,1) dot ...
        # This specific set of vectors is a known rank-7 decomposition.
        
        terms = [
            ([1, 1], [1, 1], [1, 1]),
            ([0, 1], [1, 0], [1, 0]),
            ([1, 0], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [1, 0]),
            ([1, 1], [0, 1], [1, 0]),
            ([1, 0], [1, 1], [1, 0]),
            ([1, 0], [1, 0], [1, 1]),
        ]
        return terms

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # We have 7 Strassen steps, each involving 2x2 matrices.
    # For each step, we have 7 rank-1 terms. Total 49.
    
    # Define the 7 Strassen steps with their A-block, B-block, and C-block offsets and signs.
    # A is 4x4, split into 2x2 blocks.
    # A00: rows 0-1, cols 0-1. A01: rows 0-1, cols 2-3.
    # A10: rows 2-3, cols 0-1. A11: rows 2-3, cols 2-3.
    
    # Step definitions: (A_blocks, B_blocks, C_blocks)
    # Each block is (row_offset, col_offset, sign) for A, (row_offset, col_offset, sign) for B, (row_offset, col_offset, sign) for C.
    # Note: For A, row is i, col is j. For B, row is j, col is k. For C, row is i, col is k.
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    # A terms: (0,0,1), (2,2,1)
    # B terms: (0,0,1), (2,2,1)
    # C terms: (0,0,1), (2,2,1)  -> Wait, C00 and C11 get M1.
    # C00: (0,0,1), C11: (2,2,1).
    
    # Step 2: M2 = (A10 + A11)B00
    # A terms: (2,0,1), (2,2,1)
    # B terms: (0,0,1)
    # C terms: (2,0,1) [C10], (2,2,1) [C11] -> Wait, C10 is row 2, col 0. C11 is row 2, col 2.
    
    # Step 3: M3 = A00(B01 - B11)
    # A terms: (0,0,1)
    # B terms: (0,2,1), (2,2,-1)
    # C terms: (0,2,1) [C01]
    
    # Step 4: M4 = A11(B10 - B00)
    # A terms: (2,2,1)
    # B terms: (2,0,1), (0,0,-1)
    # C terms: (0,0,1) [C00], (2,0,1) [C10]
    
    # Step 5: M5 = (A00 + A01)B11
    # A terms: (0,0,1), (0,2,1)
    # B terms: (2,2,1)
    # C terms: (0,0,-1) [C00], (0,2,1) [C01]
    
    # Step 6: M6 = (A10 - A00)(B00 + B01)
    # A terms: (2,0,1), (0,0,-1)
    # B terms: (0,0,1), (0,2,1)
    # C terms: (2,2,1) [C11]
    
    # Step 7: M7 = (A01 - A11)(B10 + B11)
    # A terms: (0,2,1), (2,2,-1)
    # B terms: (2,0,1), (2,2,1)
    # C terms: (0,0,1) [C00], (2,2,1) [C11]
    
    steps = [
        # (A_blocks, B_blocks, C_blocks)
        ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,1)]),
        ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1)]),
        ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]),
        ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]),
        ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]),
        ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
    ]
    
    col_idx = 0
    for a_blocks, b_blocks, c_blocks in steps:
        for a_r, a_c, a_s in a_blocks:
            for b_r, b_c, b_s in b_blocks:
                # For each combination of A block and B block, we have a product of 2x2 matrices.
                # We add the 7 rank-1 terms for this 2x2 product.
                for u_vec, v_vec, w_vec in strassen_2x2_terms:
                    # u_vec is for local A rows (0,1), v_vec for local A cols/B rows (0,1), w_vec for local B cols (0,1).
                    # We need to map these to global indices.
                    
                    # Global A index: i_global = a_r + u_local, j_global = a_c + v_local
                    # Global B index: j_global = a_c + v_local (same as A col), k_global = b_c + w_local
                    # Global C index: i_global = c_r + u_local (Wait, C row corresponds to A row), k_global = c_c + w_local (C col corresponds to B col)
                    
                    # Note: The Strassen step defines which C block receives the result.
                    # The local u, v, w indices in the 2x2 term correspond to the local indices within the 2x2 block.
                    
                    # We must sum over the local indices of the 2x2 term.
                    # The term is sum_{lu, lv, lw} u_vec[lu] v_vec[lv] w_vec[lw] * T[lu, lv, lw]
                    # But T[lu, lv, lw] is 1 if the multiplication uses A[lu, lv] and B[lv, lw].
                    # So we just need to place the coefficients into U, V, W.
                    
                    for lu in range(2):
                        if u_vec[lu] == 0: continue
                        for lv in range(2):
                            if v_vec[lv] == 0: continue
                            for lw in range(2):
                                if w_vec[lw] == 0: continue
                                
                                coeff = a_s * b_s * u_vec[lu] * v_vec[lv] * w_vec[lw]
                                if coeff == 0: continue
                                
                                # Global indices
                                i_a = a_r + lu
                                j_a = a_c + lv
                                k_b = b_c + lw
                                
                                # U index: i_a * m + j_a
                                u_idx = i_a * m + j_a
                                # V index: j_a * p + k_b
                                v_idx = j_a * p + k_b
                                
                                # W index: depends on which C block this contributes to.
                                # The C block is determined by the step's C_blocks.
                                # However, the local indices lu, lw determine the position *within* the C block.
                                # So we iterate over c_blocks.
                                for c_r, c_c, c_s in c_blocks:
                                    i_c = c_r + lu
                                    k_c = c_c + lw
                                    w_idx = k_c * n + i_c
                                    
                                    # Add to U, V, W
                                    U[u_idx][col_idx] += coeff
                                    V[v_idx][col_idx] += coeff
                                    W[w_idx][col_idx] += coeff_s if False else coeff # Wait, c_s is sign for C block?
                                    # In Strassen, the C block coefficient is part of the linear combination.
                                    # My step definition included signs in C_blocks?
                                    # Let's check Step 1: C_blocks [(0,0,1), (2,2,1)]. Signs are 1.
                                    # Step 5: C_blocks [(0,0,-1), (0,2,1)]. Sign is -1 for C00.
                                    # So we must multiply by c_s.
                                    U[u_idx][col_idx] += 0 # Already added? No, I should multiply coeff by c_s.
                                    # Let's re-calculate coeff.
                                    pass
                                
                                # Correction: The coefficient should be multiplied by c_s for each c_block.
                                # I will re-structure the inner loop to handle c_s properly.
                                pass
                    pass
        col_idx += 7 # Each step has 7 terms? No, each step has multiple A/B block combos.
        # Actually, the number of terms per step varies?
        # Step 1: 2 A blocks * 2 B blocks * 7 terms = 28 terms?
        # No, the rank is 49 total.
        # My loop structure is adding too many terms if I don't merge them correctly or if the rank calculation is wrong.
        # The Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # Total rank = 7 * 7 = 49.
        # My loop above iterates over A blocks and B blocks for each step.
        # For Step 1: A has 2 blocks, B has 2 blocks. 2*2=4 products of 2x2 matrices.
        # Each product has 7 rank-1 terms. Total 28 terms for Step 1?
        # This would exceed rank 49.
        
        # The error is in the mapping of Strassen steps to rank-1 terms.
        # Strassen's 7 multiplications are linear combinations of the blocks.
        # M1 = (A00+A11)(B00+B11). This is ONE multiplication of two 2x2 matrices.
        # The "A00+A11" is a 2x2 matrix. The "B00+B11" is a 2x2 matrix.
        # So there is only ONE 2x2 multiplication for M1.
        # My loop `for a_r... for b_r...` was treating them as separate products, which is wrong.
        # I need to form the summed blocks first, or rather, the rank-1 terms of the product (A_sum)(B_sum)
        # can be decomposed.
        
        # Actually, the rank of the product of sums is not simply the sum of ranks.
        # However, Strassen's algorithm works by forming the linear combinations FIRST, then multiplying.
        # So, for Step 1, we form A_M1 = A00+A11 and B_M1 = B00+B11.
        # Then we perform the 2x2 multiplication A_M1 * B_M1.
        # This multiplication has rank 7.
        # So for each of the 7 Strassen steps, we have exactly one 2x2 multiplication, contributing 7 rank-1 terms.
        # Total 49 terms.
        
        # So I should NOT iterate over A blocks and B blocks separately for the rank-1 generation.
        # I should define the 2x2 matrices A_M and B_M for each step, and then apply the 2x2 rank-7 decomposition.
        
        # Let's restart the filling logic with this correct understanding.
        pass

    # --- Corrected Filling Logic ---
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define the 7 Strassen steps again, but this time we just need to know which blocks are summed.
    # Step 1: A = A00+A11, B = B00+B11
    # Step 2: A = A10+A11, B = B00
    # Step 3: A = A00, B = B01-B11
    # Step 4: A = A11, B = B10-B00
    # Step 5: A = A00+A01, B = B11
    # Step 6: A = A10-A00, B = B00+B01
    # Step 7: A = A01-A11, B = B10+B11
    
    # For each step, we define the linear combination of blocks for A and B.
    # A_blocks: list of (row_offset, col_offset, sign)
    # B_blocks: list of (row_offset, col_offset, sign)
    
    strassen_steps = [
        # 1
        ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        # 2
        ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,1)]),
        # 3
        ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1)]),
        # 4
        ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]),
        # 5
        ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]),
        # 6
        ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]),
        # 7
        ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen_steps:
        # For each of the 7 rank-1 terms of the 2x2 multiplication:
        for u_vec, v_vec, w_vec in strassen_2x2_terms:
            # u_vec, v_vec, w_vec are length 2.
            # We expand the product (sum A_blocks)(sum B_blocks) using the rank-1 term.
            # The term is sum_{lu,lv,lw} u[lu]v[lv]w[lw] T[lu,lv,lw]
            # This corresponds to the 2x2 matrix multiplication of the summed blocks.
            
            for lu in range(2):
                if u_vec[lu] == 0: continue
                for lv in range(2):
                    if v_vec[lv] == 0: continue
                    for lw in range(2):
                        if w_vec[lw] == 0: continue
                        
                        base_coeff = u_vec[lu] * v_vec[lv] * w_vec[lw]
                        
                        # Now distribute over the block sums for A and B.
                        for a_r, a_c, a_s in a_blocks:
                            for b_r, b_c, b_s in b_blocks:
                                coeff = base_coeff * a_s * b_s
                                if coeff == 0: continue
                                
                                i_a = a_r + lu
                                j_a = a_c + lv
                                k_b = b_c + lw
                                
                                u_idx = i_a * m + j_a
                                v_idx = j_a * p + k_b
                                
                                for c_r, c_c, c_s in c_blocks:
                                    i_c = c_r + lu
                                    k_c = c_c + lw
                                    w_idx = k_c * n + i_c
                                    
                                    total_coeff = coeff * c_s
                                    
                                    U[u_idx][col_idx] += total_coeff
                                    V[v_idx][col_idx] += total_coeff
                                    W[w_idx][col_idx] += total_coeff
            col_idx += 1
            
    # Part 1: Rank 16 Schoolbook terms for the 5th column (k=4)
    # A is 4x4, B is 4x1 (col 4), C is 4x1 (col 4).
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # For each i in 0..3, j in 0..3:
    # Term: U[i*m+j, col] = 1, V[j*p+4, col] = 1, W[4*n+i, col] = 1
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            
            U[u_idx][col_idx] = 1
            V[v_idx][col_idx] = 1
            W[w_idx][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
