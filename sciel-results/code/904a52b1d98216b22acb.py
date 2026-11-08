"""Rewritten program using Strassen decomposition for the 4x4x4 core and scalar terms for the 4x4x1 remainder."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # --- Part 1: 4x4x4 Core using Strassen ---
    # The 4x4 multiplication is decomposed into 7 products of 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 scalar-like products (actually 2x2x2 tensors).
    # Total rank for core = 7 * 7 = 49.
    
    # We represent the 4x4 matrices A (4x4) and B (4x4) as blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where A11 is A[0:2, 0:2], etc.
    # B = [[B11, B12], [B21, B22]]
    
    # The 7 Strassen products for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we apply this recursively.
    # Let's define the coefficients for the 7 Strassen steps on 2x2 blocks.
    # Each step combines 2x2 sub-matrices.
    
    # Instead of recursive code which might be slow or complex to index, 
    # we can pre-calculate the 49 terms by expanding the recursion.
    
    # A 2x2 Strassen decomposition has rank 7.
    # The terms for a single 2x2x2 tensor (indices i,j,k in 0..1) are:
    # We need to map the 7 products M1..M7 to the 4 output entries C11, C12, C21, C22.
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M5
    
    # However, we are doing 4x4.
    # Let's denote the 2x2 blocks of A as A[0,0], A[0,1], A[1,0], A[1,1] (each 2x2).
    # And B similarly.
    # The 7 products M1..M7 are 2x2 matrix multiplications.
    # Each Mq is a 2x2x2 tensor of rank 7.
    # So for each q in 1..7, we have 7 sub-terms. Total 49.
    
    # Let's generate the 49 terms systematically.
    # For each of the 7 Strassen products (q=0..6):
    #   The product is of two 2x2 matrices.
    #   Let's say Product_q = X_q * Y_q.
    #   X_q and Y_q are linear combinations of the 2x2 blocks of A and B.
    #   Then we decompose X_q * Y_q (which is a 2x2x2 multiplication) using Strassen again (rank 7).
    
    # Coefficients for Strassen on 2x2 matrices (indices 0,1 for rows/cols):
    # M1: X = A00+A11, Y = B00+B11
    # M2: X = A10+A11, Y = B00
    # M3: X = A00,     Y = B01-B11
    # M4: X = A11,     Y = B10-B00
    # M5: X = A00+A01, Y = B11
    # M6: X = A10-A00, Y = B00+B01
    # M7: X = A01-A11, Y = B10+B11
    
    # We define the 7 steps. For each step, we have a linear combination of A-blocks and B-blocks.
    # Let's represent A-blocks as A_00, A_01, A_10, A_11 corresponding to indices (0,0), (0,1), (1,0), (1,1) in the 2x2 block structure.
    # In the full 4x4 matrix, A_00 is rows 0-1, cols 0-1. A_01 is rows 0-1, cols 2-3. etc.
    
    # We will iterate through the 7 outer Strassen steps.
    # For each outer step q, we determine the coefficients for A-blocks and B-blocks.
    # Then we iterate through the 7 inner Strassen steps (for the 2x2 multiplication of the combined blocks).
    
    outer_steps = [
        # (coeff_A_blocks, coeff_B_blocks)
        # A_blocks: 0=A00, 1=A01, 2=A10, 3=A11
        # B_blocks: 0=B00, 1=B01, 2=B10, 3=B11
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1: (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2: (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1]),# M3: A00(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1]),# M4: A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5: (A00+A01)B11
        ([0, 0, 1, -1], [1, 1, 0, 0]),# M6: (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1]) # M7: (A01-A11)(B10+B11)
    ]
    
    # Inner Strassen steps for 2x2 multiplication.
    # These define how to multiply two 2x2 matrices X and Y.
    # X and Y are 2x2 matrices.
    # The 7 products are linear combinations of X's and Y's 2x2 entries? No, X and Y are the 2x2 blocks from the outer step.
    # Wait, the outer step produces 2x2 matrices X and Y.
    # The multiplication X*Y is a 2x2x2 tensor.
    # We decompose X*Y using Strassen.
    # The 7 products for X*Y are:
    # N1 = (X00+X11)(Y00+Y11)
    # N2 = (X10+X11)Y00
    # N3 = X00(Y01-Y11)
    # N4 = X11(Y10-Y00)
    # N5 = (X00+X01)Y11
    # N6 = (X10-X00)(Y00+Y01)
    # N7 = (X01-X11)(Y10+Y11)
    
    # Here X00, X01, X10, X11 are the entries of the 2x2 matrix X.
    # But X itself is a linear combination of A-blocks.
    # So we need to substitute the coefficients.
    
    # Let's refine the approach.
    # For each outer step q, we have X_q = sum(c_Ai * A_i) and Y_q = sum(c_Bj * B_j).
    # A_i are 2x2 matrices.
    # X_q is a 2x2 matrix. Let's say X_q = [[x00, x01], [x10, x11]].
    # x00 is a linear combination of the (0,0) entries of the A-blocks.
    # Specifically, if A_i is the i-th 2x2 block of A, then x00 = sum(c_Ai * A_i[0,0]).
    # Similarly for x01, x10, x11.
    
    # Then we compute the 7 inner Strassen products for X_q * Y_q.
    # Each inner product N_k is a scalar (since X_q and Y_q are 2x2, their product entries are scalars? No, A_i are 2x2, so X_q is 2x2. The product X_q * Y_q is 2x2. The entries of X_q * Y_q are linear combinations of products of entries of X_q and Y_q.
    # Wait, the "rank 7" Strassen algorithm multiplies two 2x2 matrices. The result is a 2x2 matrix.
    # The 7 "multiplications" are scalar multiplications of the entries of the 2x2 matrices X_q and Y_q?
    # No. Strassen multiplies 2x2 matrices. The inputs are 2x2 matrices. The "scalar multiplications" in the Strassen algorithm refer to the multiplications of the linear combinations of the *entries* of the input matrices.
    # So, for the inner step, we are multiplying two 2x2 matrices X_q and Y_q.
    # The entries of X_q are linear combinations of the entries of the A-blocks.
    # The entries of Y_q are linear combinations of the entries of the B-blocks.
    # The 7 Strassen products N1..N7 are products of linear combinations of the *entries* of X_q and Y_q.
    # Since the entries of X_q are already linear combinations of A-entries, N1..N7 are products of linear combinations of A-entries and B-entries.
    # This matches the tensor decomposition form: sum (linear_comb_A) * (linear_comb_B) * (linear_comb_C).
    
    # Let's implement this.
    
    # Helper to get the coefficient of a specific entry (r, c) in a 2x2 block A_i
    # A_i corresponds to block (br, bc) in the 4x4 matrix.
    # The entry (r, c) in A_i corresponds to global index (br*2+r, bc*2+c).
    # In U, the row index is i_global * m + j_global.
    # i_global = br*2+r, j_global = bc*2+c.
    
    def get_U_index(br, bc, r, c):
        i_global = br * 2 + r
        j_global = bc * 2 + c
        return i_global * 4 + j_global # m=4
        
    def get_V_index(br, bc, r, c):
        # V is for B (4x4 core part, so j_global in 0..3, k_global in 0..3)
        # B is 4x4. Block (br, bc). Entry (r, c).
        j_global = br * 2 + r
        k_global = bc * 2 + c
        return j_global * 4 + k_global # p=4 for core
        
    def get_W_index(br, bc, r, c):
        # W is for C (4x4 core part, so k_global in 0..3, i_global in 0..3)
        # C is 4x4. Block (br, bc). Entry (r, c).
        k_global = br * 2 + r
        i_global = bc * 2 + c
        return k_global * 4 + i_global # n=4
        
    # We will store the terms in lists.
    core_U = []
    core_V = []
    core_W = []
    
    # 2x2 Strassen inner steps coefficients
    # Each step defines X and Y as linear combinations of X's and Y's 2x2 entries.
    # X entries: x00, x01, x10, x11
    # Y entries: y00, y01, y10, y11
    
    inner_steps = [
        # (coeffs_X, coeffs_Y)
        # coeffs_X: [c00, c01, c10, c11] for X entries
        # coeffs_Y: [c00, c01, c10, c11] for Y entries
        ([1, 0, 0, 1], [1, 0, 0, 1]), # N1: (x00+x11)(y00+y11)
        ([0, 0, 1, 1], [1, 0, 0, 0]), # N2: (x10+x11)y00
        ([1, 0, 0, 0], [0, 1, 0, -1]),# N3: x00(y01-y11)
        ([0, 0, 0, 1], [0, 0, 1, -1]),# N4: x11(y10-y00)
        ([1, 1, 0, 0], [0, 0, 0, 1]), # N5: (x00+x01)y11
        ([0, 0, 1, -1], [1, 1, 0, 0]),# N6: (x10-x00)(y00+y01)
        ([0, 1, 0, -1], [0, 0, 1, 1]) # N7: (x01-x11)(y10+y11)
    ]
    
    # For each outer step q:
    for q in range(7):
        cA_outer = outer_steps[q][0] # coeffs for A00, A01, A10, A11
        cB_outer = outer_steps[q][1] # coeffs for B00, B01, B10, B11
        
        # Construct X_q and Y_q entries as linear combinations of A and B entries.
        # X_q is 2x2. Entries x00, x01, x10, x11.
        # x00 = sum(cA_outer[i] * A_i[0,0])
        # x01 = sum(cA_outer[i] * A_i[0,1])
        # x10 = sum(cA_outer[i] * A_i[1,0])
        # x11 = sum(cA_outer[i] * A_i[1,1])
        
        # We need to represent x00 etc. as linear combinations of the 16 entries of A.
        # A has 16 entries.
        # Let's create a vector of coefficients for A for each of x00, x01, x10, x11.
        
        # Initialize coefficient vectors for X entries (size 16)
        x_coeffs = [[0.0]*16 for _ in range(4)] # 4 entries
        y_coeffs = [[0.0]*16 for _ in range(4)] # 4 entries
        
        for i_block in range(4): # 0=A00, 1=A01, 2=A10, 3=A11
            coeff = cA_outer[i_block]
            if coeff == 0: continue
            br, bc = divmod(i_block, 2)
            # A_i is at block (br, bc)
            # Entries:
            # (0,0) -> global (br*2, bc*2) -> U_idx
            # (0,1) -> global (br*2, bc*2+1)
            # (1,0) -> global (br*2+1, bc*2)
            # (1,1) -> global (br*2+1, bc*2+1)
            
            u_00 = get_U_index(br, bc, 0, 0)
            u_01 = get_U_index(br, bc, 0, 1)
            u_10 = get_U_index(br, bc, 1, 0)
            u_11 = get_U_index(br, bc, 1, 1)
            
            x_coeffs[0][u_00] += coeff
            x_coeffs[1][u_01] += coeff
            x_coeffs[2][u_10] += coeff
            x_coeffs[3][u_11] += coeff
            
            # Similarly for Y (B)
            coeff_b = cB_outer[i_block]
            if coeff_b == 0: continue
            v_00 = get_V_index(br, bc, 0, 0)
            v_01 = get_V_index(br, bc, 0, 1)
            v_10 = get_V_index(br, bc, 1, 0)
            v_11 = get_V_index(br, bc, 1, 1)
            
            y_coeffs[0][v_00] += coeff_b
            y_coeffs[1][v_01] += coeff_b
            y_coeffs[2][v_10] += coeff_b
            y_coeffs[3][v_11] += coeff_b
        
        # Now apply inner Strassen steps
        for k in range(7):
            cx = inner_steps[k][0] # coeffs for x00, x01, x10, x11
            cy = inner_steps[k][1] # coeffs for y00, y01, y10, y11
            
            # The product term is (sum cx_i * x_i) * (sum cy_j * y_j)
            # x_i is a linear combination of A entries.
            # So the total U vector is sum cx_i * x_coeffs[i]
            # Total V vector is sum cy_j * y_coeffs[j]
            
            u_term = [0.0]*16
            for i in range(4):
                if cx[i] != 0:
                    for idx in range(16):
                        u_term[idx] += cx[i] * x_coeffs[i][idx]
                        
            v_term = [0.0]*16
            for j in range(4):
                if cy[j] != 0:
                    for idx in range(16):
                        v_term[idx] += cy[j] * y_coeffs[j][idx]
                        
            # Now we need to determine the W term.
            # The inner Strassen products N1..N7 contribute to the 2x2 result C_block.
            # C_block = X_q * Y_q.
            # The 7 products N1..N7 are used to reconstruct C_block entries.
            # C00 = N1 - N4 + N6 + N7
            # C01 = N3 + N5
            # C10 = N2 + N4
            # C11 = N1 - N3 + N2 + N5
            
            # We are at inner step k (0..6).
            # We need to add the contribution of N_{k+1} to the W matrix.
            # The W matrix entries correspond to the global C matrix entries.
            # The result of X_q * Y_q is a 2x2 matrix, which is one of the 4 blocks of the 4x4 C matrix.
            # Which block?
            # The outer Strassen algorithm computes the 4 blocks of C:
            # C00 = M1 - M4 + M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M2 + M5
            
            # So, for outer step q (which is M_{q+1}), we need to know which C blocks it contributes to and with what coefficient.
            
            # Let's define the contribution of M_q to C blocks.
            # q=0 (M1): C00 (+1), C11 (+1)
            # q=1 (M2): C10 (+1), C11 (+1)
            # q=2 (M3): C01 (+1), C11 (-1)
            # q=3 (M4): C00 (-1), C10 (+1)
            # q=4 (M5): C01 (+1), C11 (+1)
            # q=5 (M6): C00 (+1)
            # q=6 (M7): C00 (+1), C11 (+1) -- Wait, let's re-verify Strassen output formulas.
            
            # Standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 - M4 + M6 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M3 + M2 + M5
            
            # My q indices:
            # q=0: M1
            # q=1: M2
            # q=2: M3
            # q=3: M4
            # q=4: M5
            # q=5: M6
            # q=6: M7
            
            # Contributions to C blocks (indices 00, 01, 10, 11):
            # M1 (q=0): C00: 1, C11: 1
            # M2 (q=1): C10: 1, C11: 1
            # M3 (q=2): C01: 1, C11: -1
            # M4 (q=3): C00: -1, C10: 1
            # M5 (q=4): C01: 1, C11: 1
            # M6 (q=5): C00: 1
            # M7 (q=6): C00: 1, C11: 1
            
            contribs = {
                0: {0: 1, 3: 1}, # M1 -> C00, C11
                1: {2: 1, 3: 1}, # M2 -> C10, C11
                2: {1: 1, 3: -1},# M3 -> C01, C11
                3: {0: -1, 2: 1},# M4 -> C00, C10
                4: {1: 1, 3: 1}, # M5 -> C01, C11
                5: {0: 1},       # M6 -> C00
                6: {0: 1, 3: 1}  # M7 -> C00, C11
            }
            
            # For the current inner step k (N_{k+1}), it contributes to the entries of the 2x2 matrix M_q.
            # Let's say M_q is a 2x2 matrix with entries m00, m01, m10, m11.
            # N1..N7 are used to compute m00..m11.
            # m00 = N1 - N4 + N6 + N7
            # m01 = N3 + N5
            # m10 = N2 + N4
            # m11 = N1 - N3 + N2 + N5
            
            # So for inner step k, we determine which m_ij it contributes to.
            inner_contribs = {
                0: {0: 1, 3: 1}, # N1 -> m00, m11
                1: {2: 1, 3: 1}, # N2 -> m10, m11
                2: {1: 1, 3: -1},# N3 -> m01, m11
                3: {0: -1, 2: 1},# N4 -> m00, m10
                4: {1: 1, 3: 1}, # N5 -> m01, m11
                5: {0: 1},       # N6 -> m00
                6: {0: 1, 3: 1}  # N7 -> m00, m11
            }
            
            w_term = [0.0]*16
            for m_idx, coeff_m in inner_contribs[k].items():
                # m_idx is 0,1,2,3 corresponding to m00, m01, m10, m11
                # The block M_q is located at... wait.
                # The outer Strassen computes the 4 blocks of C.
                # M_q is a 2x2 matrix.
                # The entries of M_q are the entries of the corresponding C block.
                # So if M_q contributes to C_block bc_idx with coefficient c_block,
                # and N_k contributes to M_q entry m_idx with coefficient c_inner,
                # then the total coefficient for the global C entry corresponding to (C_block bc_idx, entry m_idx) is c_block * c_inner.
                
                # Which C block does M_q correspond to?
                # Actually, M_q is NOT a single C block.
                # M_q is a 2x2 matrix result of multiplying two 2x2 matrices.
                # The outer Strassen formula combines these 2x2 matrices to form the final 4x4 C.
                # C00 = M1 - M4 + M6 + M7
                # This means C00 (a 2x2 block) is the sum of the 2x2 matrices M1, -M4, M6, M7.
                # So, the entry m_ij of M_q contributes to the entry (block_row, block_col, i, j) of C with the coefficient specified in `contribs[q]`.
                
                # So, for a given q and k:
                # 1. Get `coeff_block` from `contribs[q]` for each C block.
                # 2. Get `coeff_inner` from `inner_contribs[k]` for each M_q entry.
                # 3. The global C entry is determined by the block index and the entry index.
                
                for c_block_idx, coeff_block in contribs[q].items():
                    if coeff_block == 0: continue
                    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
                    c_br, c_bc = divmod(c_block_idx, 2)
                    
                    for m_entry_idx, coeff_inner in inner_contribs[k].items():
                        if coeff_inner == 0: continue
                        # m_entry_idx: 0=m00, 1=m01, 2=m10, 3=m11
                        r, c = divmod(m_entry_idx, 2)
                        
                        # Global C index:
                        # Block (c_br, c_bc), entry (r, c)
                        # k_global = c_br * 2 + r
                        # i_global = c_bc * 2 + c
                        w_idx = get_W_index(c_br, c_bc, r, c)
                        
                        w_term[w_idx] += coeff_block * coeff_inner
                        
            # Append term
            core_U.append(u_term)
            core_V.append(v_term)
            core_W.append(w_term)

    # --- Part 2: 4x4x1 Remainder ---
    # Multiply A (4x4) by B_col (4x1) to get C_col (4x1).
    # This is 4 dot products of length 4.
    # Rank = 16.
    # For each output row i (0..3) and each column k (0..3) of the remainder part (which is the 5th column of B, so k=4 in global B, but 0..3 in local remainder index? No, B is 4x5. The core used cols 0..3. The remainder is col 4.
    # Wait, the task is 4x4 * 4x5.
    # Core: 4x4 * 4x4 (cols 0-3 of B).
    # Remainder: 4x4 * 4x1 (col 4 of B).
    # Output: 4x4 (cols 0-3 of C) + 4x1 (col 4 of C).
    
    # U for remainder:
    # A is 4x4. We need to multiply by the vector b_4 (4x1).
    # The terms are A[i, j] * b_4[j] for j=0..3, summed for each i.
    # So for each i (0..3) and j (0..3), we have a term:
    # U[i*m + j] = 1 (where m=4, so U[i*4+j])
    # V[j*p + 4] = 1 (where p=5, so V[j*5+4]) -- Note: V index for B. B is 4x5. Row j, Col 4.
    # W[4*n + i] = 1 (where n=4, so W[4*4+i]) -- Note: W index for C. C is 4x5. Row 4, Col i.
    
    rem_U = []
    rem_V = []
    rem_W = []
    
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * 16
            u_vec[i * 4 + j] = 1.0
            rem_U.append(u_vec)
            
            v_vec = [0.0] * 20 # m*p = 4*5
            v_vec[j * 5 + 4] = 1.0
            rem_V.append(v_vec)
            
            w_vec = [0.0] * 20 # p*n = 5*4
            w_vec[4 * 4 + i] = 1.0
            rem_W.append(w_vec)

    # --- Combine ---
    # Core U is 16x49. Rem U is 16x16.
    # We need to construct the final matrices.
    
    total_rank = len(core_U) + len(rem_U)
    
    # Initialize final matrices
    U_final = [[0.0] * total_rank for _ in range(16)]
    V_final = [[0.0] * total_rank for _ in range(20)]
    W_final = [[0.0] * total_rank for _ in range(20)]
    
    # Fill Core
    for r in range(49):
        for i in range(16):
            if core_U[r][i] != 0:
                U_final[i][r] = core_U[r][i]
        for j in range(20):
            if core_V[r][j] != 0:
                V_final[j][r] = core_V[r][j]
        for k in range(20):
            if core_W[r][k] != 0:
                W_final[k][r] = core_W[r][k]
                
    # Fill Remainder
    offset = 49
    for r in range(16):
        for i in range(16):
            if rem_U[r][i] != 0:
                U_final[i][offset + r] = rem_U[r][i]
        for j in range(20):
            if rem_V[r][j] != 0:
                V_final[j][offset + r] = rem_V[r][j]
        for k in range(20):
            if rem_W[r][k] != 0:
                W_final[k][offset + r] = rem_W[r][k]

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
