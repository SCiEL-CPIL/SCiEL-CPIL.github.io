import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # 1. Construct 2x2x2 Strassen decomposition (Rank 7)
    # U, V, W are 4x7
    U_2x2 = np.zeros((4, 7))
    V_2x2 = np.zeros((4, 7))
    W_2x2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_2x2[:, 0] = [1, 0, 0, 1]
    V_2x2[:, 0] = [1, 0, 0, 1]
    W_2x2[:, 0] = [1, 0, 0, 1]
    
    # M2: (A10+A11)B00 -> C10, -C11
    U_2x2[:, 1] = [0, 0, 1, 1]
    V_2x2[:, 1] = [1, 0, 0, 0]
    W_2x2[:, 1] = [0, 0, 1, -1]
    
    # M3: A00(B01-B11) -> C01, C11
    U_2x2[:, 2] = [1, 0, 0, 0]
    V_2x2[:, 2] = [0, 1, 0, -1]
    W_2x2[:, 2] = [0, 1, 0, 1]
    
    # M4: A11(B10-B00) -> C00, C10
    U_2x2[:, 3] = [0, 0, 0, 1]
    V_2x2[:, 3] = [-1, 0, 1, 0]
    W_2x2[:, 3] = [1, 0, 1, 0]
    
    # M5: (A00+A01)B11 -> -C00, C01
    U_2x2[:, 4] = [1, 1, 0, 0]
    V_2x2[:, 4] = [0, 0, 0, 1]
    W_2x2[:, 4] = [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B01) -> C11
    U_2x2[:, 5] = [-1, 0, 1, 0]
    V_2x2[:, 5] = [1, 1, 0, 0]
    W_2x2[:, 5] = [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11) -> C00
    U_2x2[:, 6] = [0, 1, 0, -1]
    V_2x2[:, 6] = [0, 0, 1, 1]
    W_2x2[:, 6] = [1, 0, 0, 0]
    
    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    # U, V, W are 16x49
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    # Strassen terms for 4x4x4
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Define block indices for A, B, C
    # A: 0-3 (11), 4-7 (12), 8-11 (21), 12-15 (22)
    # B: 0-3 (11), 4-7 (12), 8-11 (21), 12-15 (22)
    # C: 0-3 (11), 4-7 (12), 8-11 (21), 12-15 (22)
    
    def add_strassen_term(U_out, V_out, W_out, col_offset, a_blocks, b_blocks, c_blocks, coeffs):
        # a_blocks: list of (block_idx, coeff)
        # b_blocks: list of (block_idx, coeff)
        # c_blocks: list of (block_idx, coeff)
        for s in range(7):
            col = col_offset + s
            u_vec = U_2x2[:, s]
            v_vec = V_2x2[:, s]
            w_vec = W_2x2[:, s]
            
            for idx, c in a_blocks:
                U_out[idx*4:(idx+1)*4, col] += c * u_vec
            for idx, c in b_blocks:
                V_out[idx*4:(idx+1)*4, col] += c * v_vec
            for idx, c in c_blocks:
                W_out[idx*4:(idx+1)*4, col] += c * w_vec

    # M1: (A11+A22)(B11+B22) -> C11, C22
    # C11 = M1 + ...
    # C22 = M1 + ...
    add_strassen_term(U_4x4, V_4x4, W_4x4, 0, 
                      [(0, 1), (3, 1)], 
                      [(0, 1), (3, 1)], 
                      [(0, 1), (3, 1)], 1)
                      
    # M2: (A21+A22)B11 -> C21, -C22
    add_strassen_term(U_4x4, V_4x4, W_4x4, 7, 
                      [(2, 1), (3, 1)], 
                      [(0, 1)], 
                      [(2, 1), (3, -1)], 1)
                      
    # M3: A11(B12-B22) -> C12, C22
    add_strassen_term(U_4x4, V_4x4, W_4x4, 14, 
                      [(0, 1)], 
                      [(1, 1), (3, -1)], 
                      [(1, 1), (3, 1)], 1)
                      
    # M4: A22(B21-B11) -> C11, C21
    add_strassen_term(U_4x4, V_4x4, W_4x4, 21, 
                      [(3, 1)], 
                      [(2, 1), (0, -1)], 
                      [(0, 1), (2, 1)], 1)
                      
    # M5: (A11+A12)B22 -> -C11, C12
    add_strassen_term(U_4x4, V_4x4, W_4x4, 28, 
                      [(0, 1), (1, 1)], 
                      [(3, 1)], 
                      [(0, -1), (1, 1)], 1)
                      
    # M6: (A21-A11)(B11+B12) -> C22
    add_strassen_term(U_4x4, V_4x4, W_4x4, 35, 
                      [(2, 1), (0, -1)], 
                      [(0, 1), (1, 1)], 
                      [(3, 1)], 1)
                      
    # M7: (A12-A22)(B21+B22) -> C11
    add_strassen_term(U_4x4, V_4x4, W_4x4, 42, 
                      [(1, 1), (3, -1)], 
                      [(2, 1), (3, 1)], 
                      [(0, 1)], 1)
                      
    # 3. Construct 4x4x3 Schoolbook decomposition (Rank 48)
    U_4x4x3 = np.zeros((16, 48))
    V_4x4x3 = np.zeros((12, 48))
    W_4x4x3 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_4x4x3[i*4+j, col] = 1
                V_4x4x3[j*3+k, col] = 1
                W_4x4x3[k*4+i, col] = 1
                col += 1
                
    # 4. Assemble 4x4x7 decomposition
    # U: 16x97
    U_total = np.zeros((16, 97))
    V_total = np.zeros((28, 97))
    W_total = np.zeros((28, 97))
    
    # Copy 4x4x4 part (cols 0-48)
    U_total[:, 0:49] = U_4x4
    V_total[0:16, 0:49] = V_4x4
    W_total[0:16, 0:49] = W_4x4
    
    # Copy 4x4x3 part (cols 49-96)
    U_total[:, 49:97] = U_4x4x3
    V_total[16:28, 49:97] = V_4x4x3
    W_total[16:28, 49:97] = W_4x4x3
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
