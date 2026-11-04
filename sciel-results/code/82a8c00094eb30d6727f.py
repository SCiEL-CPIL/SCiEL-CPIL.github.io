import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen Decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # 2. Build 4x4x4 Decomposition (Rank 49)
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    cA = np.zeros((7, 2, 2))
    cB = np.zeros((7, 2, 2))
    coeffC = np.zeros((7, 2, 2))
    
    # M1
    cA[0, 0, 0] = 1; cA[0, 1, 1] = 1
    cB[0, 0, 0] = 1; cB[0, 1, 1] = 1
    coeffC[0, 0, 0] = 1; coeffC[0, 1, 1] = 1
    
    # M2
    cA[1, 1, 0] = 1; cA[1, 1, 1] = 1
    cB[1, 0, 0] = 1
    coeffC[1, 1, 0] = 1; coeffC[1, 1, 1] = -1
    
    # M3
    cA[2, 0, 0] = 1
    cB[2, 0, 1] = 1; cB[2, 1, 1] = -1
    coeffC[2, 0, 1] = 1; coeffC[2, 1, 1] = 1
    
    # M4
    cA[3, 1, 1] = 1
    cB[3, 1, 0] = 1; cB[3, 0, 0] = -1
    coeffC[3, 0, 0] = 1; coeffC[3, 1, 0] = 1
    
    # M5
    cA[4, 0, 0] = 1; cA[4, 0, 1] = 1
    cB[4, 1, 1] = 1
    coeffC[4, 0, 0] = -1; coeffC[4, 0, 1] = 1
    
    # M6
    cA[5, 1, 0] = 1; cA[5, 0, 0] = -1
    cB[5, 0, 0] = 1; cB[5, 0, 1] = 1
    coeffC[5, 1, 1] = 1
    
    # M7
    cA[6, 0, 1] = 1; cA[6, 1, 1] = -1
    cB[6, 1, 0] = 1; cB[6, 1, 1] = 1
    coeffC[6, 0, 0] = 1
    
    for r2 in range(7):
        for r_local in range(7):
            r_global = r2 * 7 + r_local
            
            # U4
            for I in range(2):
                for J in range(2):
                    if cA[r2, I, J] != 0:
                        for i in range(2):
                            for j in range(2):
                                global_A_idx = (I*2+i)*4 + (J*2+j)
                                local_U_idx = i*2 + j
                                U4[global_A_idx, r_global] += cA[r2, I, J] * U2[local_U_idx, r_local]
                                
            # V4
            for K in range(2):
                for L in range(2):
                    if cB[r2, K, L] != 0:
                        for j in range(2):
                            for k in range(2):
                                global_B_idx = (K*2+j)*4 + (L*2+k)
                                local_V_idx = j*2 + k
                                V4[global_B_idx, r_global] += cB[r2, K, L] * V2[local_V_idx, r_local]
                                
            # W4
            for I in range(2):
                for K in range(2):
                    if coeffC[r2, I, K] != 0:
                        for i in range(2):
                            for k in range(2):
                                global_C_idx = (I*2+i)*4 + (K*2+k)
                                local_W_idx = k*2 + i
                                W4[global_C_idx, r_global] += coeffC[r2, I, K] * W2[local_W_idx, r_local]
                                
    # 3. Build 4x4x8 Decomposition (Rank 98)
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Block 1: B[:, 0:4] -> C[:, 0:4]
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, 0:49] = V4[j*4 + k_local, 0:49]
    for k_local in range(4):
        for i in range(4):
            W[k_local*4 + i, 0:49] = W4[k_local*4 + i, 0:49]
            
    # Block 2: B[:, 4:8] -> C[:, 4:8]
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + (4 + k_local), 49:98] = V4[j*4 + k_local, 0:49]
    for k_local in range(4):
        for i in range(4):
            W[(4 + k_local)*4 + i, 49:98] = W4[k_local*4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
