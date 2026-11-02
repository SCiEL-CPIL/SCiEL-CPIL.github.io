import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 6
    
    # 1. Base: 2x2x2 Strassen Decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[1, 1] = 1 # C10 (k=0, i=1 -> idx 1)
    
    # M3 = A00(B01-B11)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[2, 2] = 1 # C01 (k=1, i=0 -> idx 2)
    
    # M4 = A11(B10-B00)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[1, 3] = 1 # C10 (k=0, i=1 -> idx 1)
    
    # M5 = (A00+A01)B11
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[3, 4] = 1 # C00 (-1), C11 (1)
    
    # M6 = (A10-A00)(B00+B01)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1 # C11 (k=1, i=1 -> idx 3)
    
    # M7 = (A01-A11)(B10+B11)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1 # C00 (k=0, i=0 -> idx 0)
    
    # 2. Expand to 2x2x6
    # n=2, m=2, p=6
    # U: 4x21, V: 12x21, W: 12x21
    U_226 = np.zeros((4, 21))
    V_226 = np.zeros((12, 21))
    W_226 = np.zeros((12, 21))
    
    for b in range(3): # 3 blocks of k
        col_offset = b * 7
        # U is same
        U_226[:, col_offset:col_offset+7] = U_222
        # V: j*6 + (k_local + 2*b)
        for j in range(2):
            for k_loc in range(2):
                v_row = j * 6 + (k_loc + 2 * b)
                v_row_base = j * 2 + k_loc
                V_226[v_row, col_offset:col_offset+7] = V_222[v_row_base, :]
        # W: (k_local + 2*b)*2 + i
        for k_loc in range(2):
            for i in range(2):
                w_row = (k_loc + 2 * b) * 2 + i
                w_row_base = k_loc * 2 + i
                W_226[w_row, col_offset:col_offset+7] = W_222[w_row_base, :]
                
    # 3. Expand to 4x2x6
    # n=4, m=2, p=6
    # U: 8x42, V: 12x42, W: 24x42
    U_426 = np.zeros((8, 42))
    V_426 = np.zeros((12, 42))
    W_426 = np.zeros((24, 42))
    
    for b in range(2): # 2 blocks of i
        col_offset = b * 21
        # U: (i_local + 2*b)*2 + j
        for i_loc in range(2):
            for j in range(2):
                u_row = (i_loc + 2 * b) * 2 + j
                u_row_base = i_loc * 2 + j
                U_426[u_row, col_offset:col_offset+21] = U_226[u_row_base, :]
        # V is same
        V_426[:, col_offset:col_offset+21] = V_226
        # W: k*4 + (i_local + 2*b)
        for k in range(6):
            for i_loc in range(2):
                w_row = k * 4 + (i_loc + 2 * b)
                w_row_base = k * 2 + i_loc
                W_426[w_row, col_offset:col_offset+21] = W_226[w_row_base, :]
                
    # 4. Expand to 4x4x6
    # n=4, m=4, p=6
    # U: 16x84, V: 24x84, W: 24x84
    U_final = np.zeros((16, 84))
    V_final = np.zeros((24, 84))
    W_final = np.zeros((24, 84))
    
    for b in range(2): # 2 blocks of j
        col_offset = b * 42
        # U: i*4 + (j_local + 2*b)
        for i in range(4):
            for j_loc in range(2):
                u_row = i * 4 + (j_loc + 2 * b)
                u_row_base = i * 2 + j_loc
                U_final[u_row, col_offset:col_offset+42] = U_426[u_row_base, :]
        # V: (j_local + 2*b)*6 + k
        for j_loc in range(2):
            for k in range(6):
                v_row = (j_loc + 2 * b) * 6 + k
                v_row_base = j_loc * 6 + k
                V_final[v_row, col_offset:col_offset+42] = V_426[v_row_base, :]
        # W is same
        W_final[:, col_offset:col_offset+42] = W_426
        
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
