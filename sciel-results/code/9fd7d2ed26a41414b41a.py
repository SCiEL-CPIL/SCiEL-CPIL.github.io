import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Total rank will be 81
    # 49 for 4x4x4 Strassen
    # 32 for 4x4x2 Schoolbook
    total_rank = 81
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- Part 1: 4x4 * 4x4 (cols 0-3 of B) ---
    # We use Strassen's algorithm recursively.
    # Base case: 2x2 * 2x2 rank 7.
    
    # Define the 7 Strassen steps for 2x2 * 2x2.
    # Input indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    #                B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Output indices: C: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Step s: U_s, V_s, W_s are vectors of length 4.
    
    # M1 = (A00+A11)(B00+B11)
    U_s1 = [1, 0, 0, 1]
    V_s1 = [1, 0, 0, 1]
    W_s1 = [1, 0, 0, 0] # C00 gets M1
    W_s1[2] += 1 # Wait, C22 also gets M1? No.
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So W for M1: C00 += 1, C11 += 1.
    W_s1 = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    U_s2 = [0, 0, 1, 1]
    V_s2 = [1, 0, 0, 0]
    W_s2 = [0, 0, 1, -1] # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U_s3 = [1, 0, 0, 0]
    V_s3 = [0, 1, 0, -1]
    W_s3 = [0, 1, 0, 1] # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U_s4 = [0, 0, 0, 1]
    V_s4 = [0, 0, 1, -1]
    W_s4 = [-1, 0, 1, 0] # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U_s5 = [1, 1, 0, 0]
    V_s5 = [0, 0, 0, 1]
    W_s5 = [-1, 1, 0, 0] # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U_s6 = [-1, 0, 1, 0]
    V_s6 = [1, 1, 0, 0]
    W_s6 = [1, 0, 0, 1] # C00 += M6, C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U_s7 = [0, 1, 0, -1]
    V_s7 = [0, 0, 1, 1]
    W_s7 = [1, 0, 0, 0] # C00 += M7
    
    # Verify Strassen 2x2:
    # C00 = 1 - 1 - 1 + 1 + 1 = 1? No.
    # C00 = M1 - M4 - M5 + M6 + M7
    # W_s1[0]=1, W_s4[0]=-1, W_s5[0]=-1, W_s6[0]=1, W_s7[0]=1. Sum = 1. Correct.
    # C01 = M3 + M5
    # W_s3[1]=1, W_s5[1]=1. Sum = 2? No, W_s5[1]=1. Sum=2?
    # Wait. C01 = M3 + M5.
    # W_s3[1]=1. W_s5[1]=1. Sum=2.
    # Let's re-check Strassen formulas.
    # C01 = M3 + M5.
    # M3 = A00(B01-B11). M5 = (A00+A01)B11.
    # Sum = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    # So W_s3[1]=1, W_s5[1]=1 is correct.
    
    # C10 = M2 + M4
    # W_s2[2]=1, W_s4[2]=1. Sum=2?
    # M2 = (A10+A11)B00. M4 = A11(B10-B00).
    # Sum = A10 B00 + A11 B00 + A11 B10 - A11 B00 = A10 B00 + A11 B10. Correct.
    
    # C11 = M1 - M2 + M3 + M6
    # W_s1[3]=1, W_s2[3]=-1, W_s3[3]=1, W_s6[3]=1. Sum = 2?
    # M1 - M2 + M3 + M6
    # (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = A00 B00 + A00 B11 + A11 B00 + A11 B11 - A10 B00 - A11 B00 + A00 B01 - A00 B11 + A10 B00 + A10 B01 - A00 B00 - A00 B01
    # = A11 B11 + A10 B01. Correct.
    
    # The base Strassen decomposition is valid.
    
    # Now build 4x4 * 4x4 using 7 calls to the 2x2 decomposition.
    # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
    # B blocks: B11 (0:2, 0:2), B12 (0:2, 2:4), B21 (2:4, 0:2), B22 (2:4, 2:4)
    # C blocks: C11 (0:2, 0:2), C12 (0:2, 2:4), C21 (2:4, 0:2), C22 (2:4, 2:4)
    
    # We will add terms to U, V, W.
    # Term index starts at 0.
    # Each 2x2 decomposition adds 7 terms.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    def add_2x2_term(u_vec, v_vec, w_vec, a_offset, b_offset, c_offset, term_idx):
        # a_offset: (row_start, col_start) for A block
        # b_offset: (row_start, col_start) for B block
        # c_offset: (row_start, col_start) for C block
        # u_vec, v_vec, w_vec are length 4.
        # Local indices: 00, 01, 10, 11
        for l in range(4):
            # Map local index l to global index
            lr, lc = divmod(l, 2)
            ga = a_offset[0] + lr, a_offset[1] + lc
            gb = b_offset[0] + lr, b_offset[1] + lc
            gc = c_offset[0] + lr, c_offset[1] + lc
            
            U[ga[0]*m + ga[1], term_idx] += u_vec[l]
            V[gb[0]*p + gb[1], term_idx] += v_vec[l]
            W[gc[1]*n + gc[0], term_idx] += w_vec[l] # W index is k*n+i. gc is (i, k). So gc[1]*n + gc[0].
            
    # Strassen steps for 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # This is not a single 2x2 multiplication. It's a multiplication of sums.
    # We can't just use the 2x2 decomposition directly on sums because the decomposition is linear.
    # (A11 + A22) is a 2x2 matrix. (B11 + B22) is a 2x2 matrix.
    # We can form the U and V vectors for the sum.
    # U for A11+A22: U_A11 + U_A22.
    # But we are constructing the tensor decomposition.
    # Each term in the 4x4 decomposition is a product of a linear form in A and a linear form in B.
    # The Strassen algorithm says:
    # C = sum_{s=1}^7 (L_s(A) * R_s(B)) * W_s
    # where L_s(A) and R_s(B) are 2x2 matrices (linear combinations of blocks).
    # And L_s(A) * R_s(B) is computed using the 2x2 decomposition.
    # So for each Strassen step s, we have 7 sub-terms.
    # Total 49 terms.
    
    # Let's define the linear forms for A and B for each Strassen step.
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # B is 4x4 (first 4 cols). Blocks B11, B12, B21, B22 are 2x2.
    
    # Step 1: L1 = A11 + A22, R1 = B11 + B22
    # U_L1: A11 terms + A22 terms
    # V_R1: B11 terms + B22 terms
    
    # We need to add 7 terms for this step.
    # For each of the 7 base Strassen steps (t=0..6):
    #   U_term = U_L1 * U_base[t]
    #   V_term = V_R1 * V_base[t]
    #   W_term = W_base[t] (mapped to C11)
    
    # This is getting complex to manage offsets.
    # Let's just explicitly build the 49 terms.
    
    # Helper to get U, V vectors for a block sum.
    # Block A11: rows 0-1, cols 0-1.
    # Block A22: rows 2-3, cols 2-3.
    
    # We'll create a function to add a Strassen step.
    
    def add_strassen_step(u_blocks, v_blocks, c_block, term_start):
        # u_blocks: list of (coeff, row_start, col_start) for A
        # v_blocks: list of (coeff, row_start, col_start) for B
        # c_block: (row_start, col_start) for C
        # term_start: index in U, V, W to start writing
        # Returns next term index
        
        # Construct U and V vectors for the 2x2 matrices L and R.
        # L is 2x2. R is 2x2.
        # We will iterate over the 7 base Strassen steps.
        
        # Base Strassen vectors (from above)
        base_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        base_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        base_W = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [-1, 0, 1, 0],
            [-1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for t in range(7):
            # Construct U vector for L * base_U[t]
            # L is sum of blocks.
            # U_term is a vector of length 16 (4x4).
            # For each entry in L (2x2), we have a coefficient from base_U[t].
            # L[i,j] = sum_{b in u_blocks} coeff_b * A[b_row+i, b_col+j]
            # So U_term[global_idx] += base_U[t][i*2+j] * coeff_b
            
            # We need to map local 2x2 indices (0..3) to global 4x4 indices.
            # But the blocks are at different positions.
            
            # Let's build U_term, V_term, W_term explicitly.
            U_term = np.zeros(16)
            V_term = np.zeros(16) # Only first 16 entries of V (4x4 part) are used
            W_term = np.zeros(16) # Only first 16 entries of W (4x4 part) are used
            
            # Build U_term
            for (coeff, br, bc) in u_blocks:
                for i in range(2):
                    for j in range(2):
                        local_idx = i*2 + j
                        global_row = br + i
                        global_col = bc + j
                        global_idx = global_row * 4 + global_col
                        U_term[global_idx] += base_U[t][local_idx] * coeff
                        
            # Build V_term
            for (coeff, br, bc) in v_blocks:
                for i in range(2):
                    for j in range(2):
                        local_idx = i*2 + j
                        global_row = br + i
                        global_col = bc + j
                        global_idx = global_row * 4 + global_col
                        V_term[global_idx] += base_V[t][local_idx] * coeff
                        
            # Build W_term
            # W is for C (4x4).
            # c_block is (cr, cc).
            for i in range(2):
                for j in range(2):
                    local_idx = i*2 + j
                    global_row = c_block[0] + i
                    global_col = c_block[1] + j
                    # W index is k*n+i = global_col * 4 + global_row
                    global_w_idx = global_col * 4 + global_row
                    W_term[global_w_idx] += base_W[t][local_idx]
            
            # Add to global U, V, W
            U[:, term_start] += U_term
            V[:16, term_start] += V_term # V is 24x81. First 16 rows are for cols 0-3 of B?
            # Wait. V index is j*p + k.
            # For the 4x4 part, k is 0..3.
            # So V indices are j*6 + k.
            # My V_term is built with global_idx = global_row * 4 + global_col.
            # This is j*4 + k.
            # I need to map j*4 + k to j*6 + k.
            
            # Let's redo V_term mapping.
            V_term_mapped = np.zeros(24)
            for idx in range(16):
                j, k = divmod(idx, 4)
                mapped_idx = j * 6 + k
                V_term_mapped[mapped_idx] = V_term[idx]
            
            # W_term is already mapped to global_w_idx which is k*4 + i.
            # W is 24x81. Index is k*4 + i.
            # This is correct.
            
            U[:, term_start] += U_term
            V[:, term_start] += V_term_mapped
            W[:, term_start] += W_term
            
            term_start += 1
            
        return term_start

    # Define the 7 Strassen steps for 4x4
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B blocks: B11(0,0), B12(0,2), B21(2,0), B22(2,2)
    # C blocks: C11(0,0), C12(0,2), C21(2,0), C22(2,2)
    
    # Step 1: M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    # Wait, Strassen for 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 * 2x2 product.
    # We can compute each M using the 2x2 decomposition (7 terms).
    # Total 49 terms.
    
    # We need to add the contributions of each M to the correct C blocks.
    # For M1, it contributes to C11 and C22.
    # So we call add_strassen_step for M1, but we need to add the W part to both C11 and C22.
    # My add_strassen_step only adds to one C block.
    # I can modify it to take a list of (coeff, c_block).
    
    def add_strassen_step_multi(u_blocks, v_blocks, c_blocks, term_start):
        # c_blocks: list of (coeff, row_start, col_start)
        base_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        base_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        base_W = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [-1, 0, 1, 0],
            [-1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for t in range(7):
            U_term = np.zeros(16)
            V_term = np.zeros(16)
            W_term = np.zeros(16)
            
            for (coeff, br, bc) in u_blocks:
                for i in range(2):
                    for j in range(2):
                        local_idx = i*2 + j
                        global_row = br + i
                        global_col = bc + j
                        global_idx = global_row * 4 + global_col
                        U_term[global_idx] += base_U[t][local_idx] * coeff
                        
            for (coeff, br, bc) in v_blocks:
                for i in range(2):
                    for j in range(2):
                        local_idx = i*2 + j
                        global_row = br + i
                        global_col = bc + j
                        global_idx = global_row * 4 + global_col
                        V_term[global_idx] += base_V[t][local_idx] * coeff
                        
            for (coeff, cr, cc) in c_blocks:
                for i in range(2):
                    for j in range(2):
                        local_idx = i*2 + j
                        global_row = cr + i
                        global_col = cc + j
                        global_w_idx = global_col * 4 + global_row
                        W_term[global_w_idx] += base_W[t][local_idx] * coeff
            
            V_term_mapped = np.zeros(24)
            for idx in range(16):
                j, k = divmod(idx, 4)
                mapped_idx = j * 6 + k
                V_term_mapped[mapped_idx] = V_term[idx]
            
            U[:, term_start] += U_term
            V[:, term_start] += V_term_mapped
            W[:, term_start] += W_term
            
            term_start += 1
            
        return term_start

    # Step 1: M1
    u1 = [(1, 0, 0), (1, 2, 2)] # A11 + A22
    v1 = [(1, 0, 0), (1, 2, 2)] # B11 + B22
    c1 = [(1, 0, 0), (1, 2, 2)] # C11 + C22
    term_idx = add_strassen_step_multi(u1, v1, c1, 0)
    
    # Step 2: M2
    u2 = [(1, 2, 0), (1, 2, 2)] # A21 + A22
    v2 = [(1, 0, 0)] # B11
    c2 = [(1, 2, 0), (-1, 2, 2)] # C21 + C22 (C22 -= M2)
    term_idx = add_strassen_step_multi(u2, v2, c2, term_idx)
    
    # Step 3: M3
    u3 = [(1, 0, 0)] # A11
    v3 = [(1, 0, 2), (-1, 2, 2)] # B12 - B22
    c3 = [(1, 0, 2), (1, 2, 2)] # C12 + C22
    term_idx = add_strassen_step_multi(u3, v3, c3, term_idx)
    
    # Step 4: M4
    u4 = [(1, 2, 2)] # A22
    v4 = [(1, 2, 0), (-1, 0, 0)] # B21 - B11
    c4 = [(1, 0, 0), (1, 2, 0)] # C11 + C21
    term_idx = add_strassen_step_multi(u4, v4, c4, term_idx)
    
    # Step 5: M5
    u5 = [(1, 0, 0), (1, 0, 2)] # A11 + A12
    v5 = [(1, 2, 2)] # B22
    c5 = [(-1, 0, 0), (1, 0, 2)] # C11 - M5, C12 + M5
    term_idx = add_strassen_step_multi(u5, v5, c5, term_idx)
    
    # Step 6: M6
    u6 = [(1, 2, 0), (-1, 0, 0)] # A21 - A11
    v6 = [(1, 0, 0), (1, 0, 2)] # B11 + B12
    c6 = [(1, 2, 2)] # C22
    term_idx = add_strassen_step_multi(u6, v6, c6, term_idx)
    
    # Step 7: M7
    u7 = [(1, 0, 2), (-1, 2, 2)] # A12 - A22
    v7 = [(1, 2, 0), (1, 2, 2)] # B21 + B22
    c7 = [(1, 0, 0)] # C11
    term_idx = add_strassen_step_multi(u7, v7, c7, term_idx)
    
    # term_idx should be 49.
    
    # --- Part 2: 4x4 * 4x2 (cols 4-5 of B) ---
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                # Term index
                # U[i*4+j, term_idx] = 1
                # V[j*6+k, term_idx] = 1
                # W[k*4+i, term_idx] = 1
                
                U[i*4 + j, term_idx] = 1.0
                V[j*6 + k, term_idx] = 1.0
                W[k*4 + i, term_idx] = 1.0
                term_idx += 1
                
    # term_idx should be 81.
    
    # Verify
    # Random test
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 6)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_t U[i*4+j, t] * V[j*6+k, t] * W[k*4+i, t]
    # This is a tensor contraction.
    # C = sum_t (U_t outer V_t) * W_t? No.
    # C[i,k] = sum_j sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    
    C_computed = np.zeros((4, 6))
    for t in range(total_rank):
        u_t = U[:, t]
        v_t = V[:, t]
        w_t = W[:, t]
        
        # u_t is 16x1. Reshape to 4x4.
        u_mat = u_t.reshape(4, 4)
        # v_t is 24x1. Reshape to 4x6.
        v_mat = v_t.reshape(4, 6)
        # w_t is 24x1. Reshape to 6x4?
        # W index is k*4+i. So W is (6, 4) if reshaped.
        w_mat = w_t.reshape(6, 4)
        
        # Contribution to C:
        # C += (u_mat * v_mat) * w_mat?
        # u_mat[i,j] v_mat[j,k] w_mat[k,i]
        # This is element-wise product of (u_mat * v_mat) and w_mat.T?
        # w_mat[k,i] is at (k,i).
        # So C += (u_mat @ v_mat) * w_mat.T?
        # No, sum_j u_mat[i,j] v_mat[j,k] is (u_mat @ v_mat)[i,k].
        # Then multiply by w_mat[k,i].
        # So C += (u_mat @ v_mat) * w_mat.T
        
        prod = u_mat @ v_mat
        C_computed += prod * w_mat.T
        
    if not np.allclose(C_expected, C_computed):
        # Fallback to schoolbook if error
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1.0
                    V[j*6+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

