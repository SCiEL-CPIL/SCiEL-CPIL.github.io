import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook 3x3 Decomposition (Rank 27)
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*9 + j*3 + k
                    U3[i*3+j, col] = 1.0
                    V3[j*3+k, col] = 1.0
                    W3[k*3+i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27
    
    # Strassen Formulas for 2x2 block multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Total rank R = 7 * 27 = 189
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    def add_strassen_term(start_col, 
                          a_blocks, # list of (block_idx, coeff)
                          b_blocks, # list of (block_idx, coeff)
                          c_blocks): # list of (block_idx, coeff)
        # Block indices 0..3 for the 2x2 grid:
        # 0: (0,0) top-left
        # 1: (0,1) top-right
        # 2: (1,0) bottom-left
        # 3: (1,1) bottom-right
        
        for r in range(r3):
            # Extract the 3x3 basis vectors
            u_vec = U3[:, r]
            v_vec = V3[:, r]
            w_vec = W3[:, r]
            
            # Reshape to 3x3 for easier indexing
            u_mat = u_vec.reshape(3, 3)
            v_mat = v_vec.reshape(3, 3)
            w_mat = w_vec.reshape(3, 3)
            
            # Update U6
            for bi, coeff in a_blocks:
                row_block = bi // 2
                col_block = bi % 2
                
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            global_i = row_block * 3 + i
                            global_j = col_block * 3 + j
                            global_u_idx = global_i * 6 + global_j
                            U6[global_u_idx, start_col + r] += val
            
            # Update V6
            for bi, coeff in b_blocks:
                row_block = bi // 2
                col_block = bi % 2
                
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            global_j = row_block * 3 + j
                            global_k = col_block * 3 + k
                            global_v_idx = global_j * 6 + global_k
                            V6[global_v_idx, start_col + r] += val
            
            # Update W6
            for bi, coeff in c_blocks:
                # W index is k*n + i where k is column, i is row
                row_block = bi // 2
                col_block = bi % 2
                
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            global_k = col_block * 3 + k
                            global_i = row_block * 3 + i
                            global_w_idx = global_k * 6 + global_i
                            W6[global_w_idx, start_col + r] += val

    # Strassen Formulas:
    # P1 = (A11 + A22) * (B11 + B22)
    add_strassen_term(0, 
                      a_blocks=[(0, 1), (3, 1)], 
                      b_blocks=[(0, 1), (3, 1)], 
                      c_blocks=[(0, 1), (3, 1)])
    
    # P2 = (A21 + A22) * B11
    add_strassen_term(r3, 
                      a_blocks=[(2, 1), (3, 1)], 
                      b_blocks=[(0, 1)], 
                      c_blocks=[(2, 1), (3, -1)])
    
    # P3 = A11 * (B12 - B22)
    add_strassen_term(2*r3, 
                      a_blocks=[(0, 1)], 
                      b_blocks=[(1, 1), (3, -1)], 
                      c_blocks=[(1, 1), (3, 1)])
    
    # P4 = A22 * (B21 - B11)
    add_strassen_term(3*r3, 
                      a_blocks=[(3, 1)], 
                      b_blocks=[(2, 1), (0, -1)], 
                      c_blocks=[(0, 1), (2, 1)])
    
    # P5 = (A11 + A12) * B22
    add_strassen_term(4*r3, 
                      a_blocks=[(0, 1), (1, 1)], 
                      b_blocks=[(3, 1)], 
                      c_blocks=[(1, 1), (0, -1)])
    
    # P6 = (A21 - A11) * (B11 + B12)
    add_strassen_term(5*r3, 
                      a_blocks=[(2, 1), (0, -1)], 
                      b_blocks=[(0, 1), (1, 1)], 
                      c_blocks=[(3, 1)])
    
    # P7 = (A12 - A22) * (B21 + B22)
    add_strassen_term(6*r3, 
                      a_blocks=[(1, 1), (3, -1)], 
                      b_blocks=[(2, 1), (3, 1)], 
                      c_blocks=[(0, 1)])

    # Verify the decomposition
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Should be 1 for all i, j, k.
    
    if time.time() > deadline:
        # Return without verification if we're over deadline
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    U3_reshaped = U6.reshape(6, 6, -1)
    V3_reshaped = V6.reshape(6, 6, -1)
    W3_reshaped = W6.reshape(6, 6, -1)
    
    try:
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U3_reshaped, V3_reshaped, W3_reshaped)
        target = np.ones((6, 6, 6))
        if np.allclose(reconstructed, target, atol=1e-9):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        else:
            # Fallback to schoolbook if Strassen construction failed
            rank = 216
            U_f = np.zeros((36, rank))
            V_f = np.zeros((36, rank))
            W_f = np.zeros((36, rank))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i*6 + j
                        jk = j*6 + k
                        ki = k*6 + i
                        col = i*36 + j*6 + k
                        U_f[idx, col] = 1.0
                        V_f[jk, col] = 1.0
                        W_f[ki, col] = 1.0
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    except:
        # Fallback to schoolbook
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
