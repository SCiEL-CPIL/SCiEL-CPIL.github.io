import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will build the decomposition for 4x4x9
    # Split B into three parts: B1 (4x4), B2 (4x4), B3 (4x1)
    # C1 = A B1, C2 = A B2, C3 = A B3
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x4 using Strassen (Rank 49)
    # Part 3: 4x4x1 using Schoolbook (Rank 16)
    
    # Total Rank = 49 + 49 + 16 = 114
    
    U_list = []
    V_list = []
    W_list = []
    
    # Generate Strassen 4x4x4 decomposition
    def get_strassen_4x4x4():
        U_222 = np.zeros((4, 7))
        V_222 = np.zeros((4, 7))
        W_222 = np.zeros((4, 7))
        
        # M1: a(e+h)
        U_222[0, 0] = 1; V_222[0, 0] = 1; V_222[2, 0] = 1
        W_222[0, 0] = 0; W_222[1, 0] = 1
        # M2: (a+c)h
        U_222[0, 1] = 1; U_222[2, 1] = 1; V_222[2, 1] = 1
        W_222[0, 1] = -1; W_222[1, 1] = 1
        # M3: (b+d)g
        U_222[1, 2] = 1; U_222[3, 2] = 1; V_222[1, 2] = 1
        W_222[2, 2] = 1
        # M4: d(f-g)
        U_222[3, 3] = 1; V_222[1, 3] = 1; V_222[3, 3] = -1
        W_222[0, 3] = 1; W_222[2, 3] = 1
        # M5: (a+d)(e+h)
        U_222[0, 4] = 1; U_222[3, 4] = 1; V_222[0, 4] = 1; V_222[2, 4] = 1
        W_222[0, 4] = 1; W_222[3, 4] = 1
        # M6: (b-d)(g+h)
        U_222[1, 5] = 1; U_222[3, 5] = -1; V_222[1, 5] = 1; V_222[3, 5] = 1
        W_222[0, 5] = 1
        # M7: (a-c)(e+f)
        U_222[0, 6] = 1; U_222[2, 6] = -1; V_222[0, 6] = 1; V_222[1, 6] = 1
        W_222[3, 6] = -1
        
        # Correct W mapping for Strassen:
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M2
        # C10 = M3 + M4
        # C11 = M5 - M1 + M3 - M7
        
        W_222 = np.zeros((4, 7))
        W_222[0, 1] = -1; W_222[0, 3] = 1; W_222[0, 4] = 1; W_222[0, 5] = 1
        W_222[1, 0] = 1; W_222[1, 1] = 1
        W_222[2, 2] = 1; W_222[2, 3] = 1
        W_222[3, 0] = -1; W_222[3, 2] = 1; W_222[3, 4] = 1; W_222[3, 6] = -1
        
        U_444 = np.zeros((16, 49))
        V_444 = np.zeros((16, 49))
        W_444 = np.zeros((16, 49))
        
        col_offset = 0
        for s in range(7):
            A_blocks = []
            B_blocks = []
            
            if s == 0:
                A_blocks = [(0, 1)]
                B_blocks = [(0, 1), (2, 1)]
            elif s == 1:
                A_blocks = [(0, 1), (2, 1)]
                B_blocks = [(2, 1)]
            elif s == 2:
                A_blocks = [(1, 1), (3, 1)]
                B_blocks = [(3, 1)]
            elif s == 3:
                A_blocks = [(3, 1)]
                B_blocks = [(1, 1), (3, -1)]
            elif s == 4:
                A_blocks = [(0, 1), (3, 1)]
                B_blocks = [(0, 1), (3, 1)]
            elif s == 5:
                A_blocks = [(1, 1), (3, -1)]
                B_blocks = [(2, 1), (3, 1)]
            elif s == 6:
                A_blocks = [(0, 1), (2, -1)]
                B_blocks = [(0, 1), (1, 1)]
                
            for k in range(7):
                for a_block, sign in A_blocks:
                    base_idx = [0, 2, 8, 10][a_block]
                    for local_idx in range(4):
                        global_idx = [base_idx, base_idx+1, base_idx+4, base_idx+5][local_idx]
                        U_444[global_idx, col_offset + k] += U_222[local_idx, k] * sign
                        
                for b_block, sign in B_blocks:
                    base_idx = [0, 2, 8, 10][b_block]
                    for local_idx in range(4):
                        global_idx = [base_idx, base_idx+1, base_idx+4, base_idx+5][local_idx]
                        V_444[global_idx, col_offset + k] += V_222[local_idx, k] * sign
                        
                for c_block in range(4):
                    base_idx = [0, 2, 8, 10][c_block]
                    for local_idx in range(4):
                        global_idx = [base_idx, base_idx+1, base_idx+4, base_idx+5][local_idx]
                        W_444[global_idx, col_offset + k] += W_222[local_idx, k]
                        
            col_offset += 7
            
        return U_444, V_444, W_444

    U_s, V_s, W_s = get_strassen_4x4x4()
    
    # Add Part 1
    U_list.append(U_s)
    V_full = np.zeros((36, 49))
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, :] = V_s[j*4+k, :]
    V_list.append(V_full)
    W_full = np.zeros((36, 49))
    for i in range(4):
        for k in range(4):
            W_full[k*4+i, :] = W_s[k*4+i, :]
    W_list.append(W_full)
    
    # Add Part 2
    U_list.append(U_s)
    V_full = np.zeros((36, 49))
    for j in range(4):
        for k in range(4):
            V_full[j*9+(k+4), :] = V_s[j*4+k, :]
    V_list.append(V_full)
    W_full = np.zeros((36, 49))
    for i in range(4):
        for k in range(4):
            W_full[(k+4)*4+i, :] = W_s[k*4+i, :]
    W_list.append(W_full)
    
    # Add Part 3 (4x4x1)
    U_1 = np.zeros((16, 16))
    V_1 = np.zeros((36, 16))
    W_1 = np.zeros((36, 16))
    for i in range(4):
        for j in range(4):
            idx = i*4+j
            U_1[i*4+j, idx] = 1
            V_1[j*9+8, idx] = 1
            W_1[8*4+i, idx] = 1
            
    U_list.append(U_1)
    V_list.append(V_1)
    W_list.append(W_1)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
