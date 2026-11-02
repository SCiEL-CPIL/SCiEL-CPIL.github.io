import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank will be sum of ranks of sub-problems
    # Part 1: 2x2x2 Strassen x 3 (for p=6 split into 3 pairs)
    # Part 2: 2x3x2 Schoolbook x 3 (for p=6 split into 3 pairs)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block):
        U_blocks.append(np.array(U_block))
        V_blocks.append(np.array(V_block))
        W_blocks.append(np.array(W_block))
        
    # Part 1: A1 (2x2) * B1 (2x6)
    # B1 is split into 3 matrices of 2x2
    for k_block in range(3):
        # Strassen 2x2x2
        # U is 4x7, V is 4x7, W is 4x7
        # Indices for U: i*2+j. i in {0,1}, j in {0,1}
        # Indices for V: j*2+k. j in {0,1}, k in {0,1} (local k)
        # Indices for W: k*2+i. k in {0,1} (local k), i in {0,1}
        
        # Strassen factors for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        U_str = np.zeros((4, 7))
        V_str = np.zeros((4, 7))
        W_str = np.zeros((4, 7))
        
        # c=0: M1
        U_str[0, 0] = 1; U_str[3, 0] = 1 # A00, A11
        V_str[0, 0] = 1; V_str[3, 0] = 1 # B00, B11
        W_str[0, 0] = 1; W_str[1, 0] = 1; W_str[2, 0] = 1; W_str[3, 0] = 1 # C00, C01, C10, C11
        # W indices: k*2+i. 
        # C00 is k=0,i=0 -> 0. C01 is k=1,i=0 -> 2. C10 is k=0,i=1 -> 1. C11 is k=1,i=1 -> 3.
        # So W_str[0,0]=1, W_str[2,0]=1, W_str[1,0]=1, W_str[3,0]=1.
        
        # c=1: M2
        U_str[2, 1] = 1; U_str[3, 1] = 1 # A10, A11
        V_str[0, 1] = 1 # B00
        W_str[1, 1] = 1; W_str[3, 1] = 1 # C10, C11 (M2 contributes to C10 and C11? C10=M2+M4, C11=M1-M2+M3+M6)
        # C10 is index 1. C11 is index 3.
        
        # c=2: M3
        U_str[0, 2] = 1 # A00
        V_str[1, 2] = 1; V_str[3, 2] = -1 # B01, -B11
        W_str[2, 2] = 1; W_str[3, 2] = 1 # C01, C11
        
        # c=3: M4
        U_str[3, 3] = 1 # A11
        V_str[2, 3] = 1; V_str[0, 3] = -1 # B10, -B00
        W_str[0, 3] = 1; W_str[1, 3] = 1 # C00, C10
        
        # c=4: M5
        U_str[0, 4] = 1; U_str[1, 4] = 1 # A00, A01
        V_str[3, 4] = 1 # B11
        W_str[0, 4] = -1; W_str[2, 4] = 1 # C00 (-M5), C01 (M5)
        
        # c=5: M6
        U_str[2, 5] = 1; U_str[0, 5] = -1 # A10, -A00
        V_str[0, 5] = 1; V_str[1, 5] = 1 # B00, B01
        W_str[3, 5] = 1 # C11 (M6)
        
        # c=6: M7
        U_str[1, 6] = 1; U_str[3, 6] = -1 # A01, -A11
        V_str[2, 6] = 1; V_str[3, 6] = 1 # B10, B11
        W_str[0, 6] = 1 # C00 (M7)
        
        # Map local indices to global indices
        # U global: i*5 + j. j in {0,1}.
        # V global: j*6 + k. j in {0,1}. k in {2*k_block, 2*k_block+1}.
        # W global: k*2 + i. k in {2*k_block, 2*k_block+1}.
        
        U_glob = np.zeros((10, 7))
        V_glob = np.zeros((30, 7))
        W_glob = np.zeros((12, 7))
        
        for i in range(2):
            for j in range(2):
                U_glob[i*5 + j, :] = U_str[i*2 + j, :]
                
        for j in range(2):
            for k_loc in range(2):
                k_glob = 2*k_block + k_loc
                V_glob[j*6 + k_glob, :] = V_str[j*2 + k_loc, :]
                
        for k_loc in range(2):
            k_glob = 2*k_block + k_loc
            for i in range(2):
                W_glob[k_glob*2 + i, :] = W_str[k_loc*2 + i, :]
                
        add_block(U_glob, V_glob, W_glob)
        
    # Part 2: A2 (2x3) * B2 (3x6)
    # B2 is split into 3 matrices of 3x2
    for k_block in range(3):
        # Schoolbook 2x3x2
        # Rank 12
        U_sch = np.zeros((6, 12)) # 2*3
        V_sch = np.zeros((6, 12)) # 3*2
        W_sch = np.zeros((4, 12)) # 2*2
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U_sch[i*3 + j, col] = 1
                    V_sch[j*2 + k, col] = 1
                    W_sch[k*2 + i, col] = 1
                    col += 1
                    
        # Map to global
        U_glob = np.zeros((10, 12))
        V_glob = np.zeros((30, 12))
        W_glob = np.zeros((12, 12))
        
        # A2 is j in {2,3,4}
        for i in range(2):
            for j_loc in range(3):
                j_glob = 2 + j_loc
                U_glob[i*5 + j_glob, :] = U_sch[i*3 + j_loc, :]
                
        # B2 is j in {2,3,4}, k in {2*k_block, 2*k_block+1}
        for j_loc in range(3):
            j_glob = 2 + j_loc
            for k_loc in range(2):
                k_glob = 2*k_block + k_loc
                V_glob[j_glob*6 + k_glob, :] = V_sch[j_loc*2 + k_loc, :]
                
        # W is k in {2*k_block, 2*k_block+1}, i in {0,1}
        for k_loc in range(2):
            k_glob = 2*k_block + k_loc
            for i in range(2):
                W_glob[k_glob*2 + i, :] = W_sch[k_loc*2 + i, :]
                
        add_block(U_glob, V_glob, W_glob)
        
    # Combine blocks
    U = np.concatenate(U_blocks, axis=1)
    V = np.concatenate(V_blocks, axis=1)
    W = np.concatenate(W_blocks, axis=1)
    
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
