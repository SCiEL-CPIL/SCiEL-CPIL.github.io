"""Program to find a low-rank decomposition for 4x4 @ 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def get_strassen_step():
    """
    Returns coefficients for Strassen's algorithm.
    Returns:
        U_coeff: List of 7 matrices (2x2) or coefficients for A
        V_coeff: List of 7 matrices (2x2) or coefficients for B
        W_coeff: List of 7 matrices (2x2) or coefficients for C
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    
    # A coefficients for each of the 7 terms
    # A00 is at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
    A_coefs = [
        [[1, 0], [0, 1]], # M1: A00, A11
        [[0, 0], [1, 1]], # M2: A10, A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 0], [1, 0]], # M5: A00, A10
        [[0, -1], [0, 1]], # M6: -A00, A01
        [[0, -1], [1, 0]]  # M7: -A01, A11
    ]
    
    # B coefficients for each of the 7 terms
    B_coefs = [
        [[1, 0], [0, 1]], # M1: B00, B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]], # M3: B01, -B11
        [[0, -1], [1, 0]], # M4: -B00, B10
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00, B10
        [[0, 1], [0, 1]]  # M7: B01, B11
    ]
    
    # C reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C coefficients: For each block of C, which M's contribute with what weight
    # C00
    C00 = [[1, 0], [0, 0]] # Coefs for M1..M4? No, this is for the block position in C.
    # We need W coefficients.
    # W maps from the 7 intermediate results to the 4 blocks of C.
    # Let's define W_coefs as a list of 4 matrices (2x2) corresponding to C00, C01, C10, C11.
    # Each element W_coefs[i][j] is the coefficient of M_k?
    # No, W_coefs should map the 7 M's to the C blocks.
    
    # It is easier to store W as a matrix of shape (4, 7) where W[c_block, m_idx] is the coefficient.
    # C00 uses M1(1), M4(1), M5(-1), M7(1)
    # C01 uses M3(1), M5(1)
    # C10 uses M2(1), M4(1)
    # C11 uses M1(1), M2(-1), M3(1), M6(1)
    
    C_matrix = np.zeros((4, 7), dtype=int)
    # M idx: 0..6
    C_matrix[0, 0] = 1; C_matrix[0, 3] = 1; C_matrix[0, 4] = -1; C_matrix[0, 6] = 1
    C_matrix[1, 2] = 1; C_matrix[1, 4] = 1
    C_matrix[2, 1] = 1; C_matrix[2, 3] = 1
    C_matrix[3, 0] = 1; C_matrix[3, 1] = -1; C_matrix[3, 2] = 1; C_matrix[3, 5] = 1
    
    return A_coefs, B_coefs, C_matrix

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with 1x1x1 (Rank 1)
    # U is (1x1)x1, V is (1x1)x1, W is (1x1)x1
    # Flattened: U[0][0]=1, V[0][0]=1, W[0][0]=1
    U = np.ones((1, 1, 1)) # indices: [i, j, r]
    V = np.ones((1, 1, 1)) # indices: [j, k, r]
    W = np.ones((1, 1, 1)) # indices: [k, i, r]
    
    current_n = 1
    # Target n is 4 for the square part
    while current_n < 4:
        A_c, B_c, C_mat = get_strassen_step()
        old_r = U.shape[2]
        new_r = old_r * 7
        new_n = current_n * 2
        
        # New U: shape (2n, 2n, 7*old_r)
        new_U = np.zeros((new_n, new_n, new_r), dtype=int)
        # New V: shape (2n, 2n, 7*old_r)
        new_V = np.zeros((new_n, new_n, new_r), dtype=int)
        # New W: shape (2n, 2n, 7*old_r)
        new_W = np.zeros((new_n, new_n, new_r), dtype=int)
        
        for m_idx in range(7):
            r_offset = m_idx * old_r
            # A coefficients
            a_mat = np.array(A_c[m_idx]) # 2x2
            # B coefficients
            b_mat = np.array(B_c[m_idx]) # 2x2
            # C coefficients for this block? No, W is aggregated later.
            # Actually, W depends on the final combination of M's.
            # The new W will be formed by combining the old W's with the C_matrix coefficients.
            
            # Fill new_U
            for i in range(new_n):
                for j in range(new_n):
                    ib = i // current_n
                    jb = j // current_n
                    ii = i % current_n
                    jj = j % current_n
                    coeff = a_mat[ib, jb]
                    if coeff != 0:
                        new_U[i, j, r_offset:r_offset+old_r] = coeff * U[ii, jj, :]
                        
            # Fill new_V
            for j in range(new_n):
                for k in range(new_n):
                    jb = j // current_n
                    kb = k // current_n
                    jj = j % current_n
                    kk = k % current_n
                    coeff = b_mat[jb, kb]
                    if coeff != 0:
                        new_V[j, k, r_offset:r_offset+old_r] = coeff * V[jj, kk, :]
                        
            # Fill new_W
            # The contribution of M_m_idx to the output blocks C_block.
            # C_matrix[block_idx, m_idx] is the coefficient.
            for c_block in range(4):
                coeff = C_mat[c_block, m_idx]
                if coeff != 0:
                    cb_i = c_block // 2
                    cb_j = c_block % 2
                    for k in range(new_n):
                        for i in range(new_n):
                            ib = i // current_n
                            jb = j // current_n # Wait, W is [k, i]
                            # W index is [k, i].
                            # In the sub-problem, W is [k_sub, i_sub].
                            # The block index in C is (i_block, k_block).
                            # c_block corresponds to i_block, k_block.
                            ib = i // current_n
                            kb = k // current_n
                            
                            # Wait, C_matrix maps M to C blocks.
                            # C_block index 0 -> C00 (i_block=0, k_block=0)
                            # C_block index 1 -> C01 (i_block=0, k_block=1)
                            # C_block index 2 -> C10 (i_block=1, k_block=0)
                            # C_block index 3 -> C11 (i_block=1, k_block=1)
                            
                            if ib == c_block // 2 and kb == c_block % 2:
                                ii = i % current_n
                                kk = k % current_n
                                new_W[k, i, r_offset:r_offset+old_r] = coeff * W[kk, ii, :]
                            
        U, V, W = new_U, new_V, new_W
        current_n = new_n
        if time.time() - start_time > 140:
             break

    # Now U, V, W are for 4x4x4.
    # Flatten to matrices
    U_44 = U.reshape(16, 49)
    V_44 = V.reshape(16, 49)
    W_44 = W.reshape(16, 49)
    
    # Construct for 4x4x8
    # U is 16 x 98
    # V is 32 x 98
    # W is 32 x 98
    
    U_final = np.zeros((16, 98), dtype=int)
    V_final = np.zeros((32, 98), dtype=int)
    W_final = np.zeros((32, 98), dtype=int)
    
    # Part 1: A * B[:4]
    U_final[:, 0:49] = U_44
    
    # Map V indices
    # V_44 is [j*4+k, r]
    # V_final is [j*8+k, r]
    for r in range(49):
        for j in range(4):
            for k in range(4):
                idx_local = j*4+k
                idx_global = j*8+k
                V_final[idx_global, r] = V_44[idx_local, r]
                
    # Map W indices
    # W_44 is [k*4+i, r]
    # W_final is [k*4+i, r] for k in 0..3 (first 16 rows)
    W_final[0:16, 0:49] = W_44
    
    # Part 2: A * B[4:]
    U_final[:, 49:98] = U_44
    
    for r in range(49):
        for j in range(4):
            for k in range(4):
                idx_local = j*4+k
                idx_global = j*8+(k+4)
                V_final[idx_global, r+49] = V_44[idx_local, r]
                
    # Map W indices
    # W_44 is [k*4+i, r]
    # For B[4:], the output columns are 4..7.
    # W_final rows are [k*4+i]. k in 4..7.
    # Rows 16..31.
    W_final[16:32, 49:98] = W_44
    
    # Convert to lists as required
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
