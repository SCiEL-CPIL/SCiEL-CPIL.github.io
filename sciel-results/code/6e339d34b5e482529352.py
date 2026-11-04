"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Construct M_{1,2,2} decomposition (Rank 4)
    # U_122: 1x4 (i=0, j=0,1) -> actually 2 rows for j? 
    # A is 1x2. Rows of U correspond to A entries.
    # A[0,0], A[0,1]. So U_122 has 2 rows? 
    # Wait, M_{1,2,2}: A is 1x2. U has 1*2=2 rows.
    # B is 2x2. V has 2*2=4 rows.
    # C is 1x2. W has 1*2=2 rows.
    
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    
    # Col 1: A[0,0] B[0,0] -> C[0,0]
    U_122[0, 0] = 1 # A[0,0]
    V_122[0, 0] = 1 # B[0,0]
    W_122[0, 0] = 1 # C[0,0]
    
    # Col 2: A[0,0] B[0,1] -> C[0,1]
    U_122[0, 1] = 1 # A[0,0]
    V_122[1, 1] = 1 # B[0,1]
    W_122[1, 1] = 1 # C[0,1]
    
    # Col 3: A[0,1] B[1,0] -> C[0,0]
    U_122[1, 2] = 1 # A[0,1]
    V_122[2, 2] = 1 # B[1,0]
    W_122[0, 2] = 1 # C[0,0]
    
    # Col 4: A[0,1] B[1,1] -> C[0,1]
    U_122[1, 3] = 1 # A[0,1]
    V_122[3, 3] = 1 # B[1,1]
    W_122[1, 3] = 1 # C[0,1]
    
    # Construct M_{3,2,2} decomposition (Rank 11)
    # M_{3,2,2} = M_{2,2,2} (Strassen) + M_{1,2,2} (Schoolbook)
    # U_322: 6x11
    # V_322: 4x11
    # W_322: 6x11
    
    # Strassen part (cols 0-6)
    # U_s: 4x7 (A_top 2x2)
    # V_s: 4x7 (B 2x2)
    # W_s: 4x7 (C_top 2x2)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11=U[0], A22=U[3]
    # B11=V[0], B22=V[3]
    # C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    # Col 0: M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = -1 # C11+=M1, C12+=M1, C21+=M1, C22-=M1? 
    # Wait, Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define U,V,W for each M_k
    
    # M1: (A11+A22)(B11+B22)
    U_s[:, 0] = [1, 0, 0, 1]
    V_s[:, 0] = [1, 0, 0, 1]
    
    # M2: (A21+A22)B11
    U_s[:, 1] = [0, 0, 1, 1]
    V_s[:, 1] = [1, 0, 0, 0]
    
    # M3: A11(B12-B22)
    U_s[:, 2] = [1, 0, 0, 0]
    V_s[:, 2] = [0, 1, 0, -1]
    
    # M4: A22(B21-B11)
    U_s[:, 3] = [0, 0, 0, 1]
    V_s[:, 3] = [0, 0, 1, -1]
    
    # M5: (A11+A12)B22
    U_s[:, 4] = [1, 1, 0, 0]
    V_s[:, 4] = [0, 0, 0, 1]
    
    # M6: (A21-A11)(B11+B12)
    U_s[:, 5] = [-1, 0, 1, 0]
    V_s[:, 5] = [1, 1, 0, 0]
    
    # M7: (A12-A22)(B21+B22)
    U_s[:, 6] = [0, 1, 0, -1]
    V_s[:, 6] = [0, 0, 1, 1]
    
    # Now W_s based on C formulas
    # C11 (idx 0): M1 + M4 - M5 + M7
    W_s[0, :] = [1, 0, 0, 1, -1, 0, 1]
    # C12 (idx 1): M3 + M5
    W_s[1, :] = [0, 0, 1, 0, 1, 0, 0]
    # C21 (idx 2): M2 + M4
    W_s[2, :] = [0, 1, 0, 1, 0, 0, 0]
    # C22 (idx 3): M1 - M2 + M3 + M6
    W_s[3, :] = [1, -1, 1, 0, 0, 1, 0]
    
    # Schoolbook part for M_{1,2,2} (cols 7-10)
    # A_bot is 1x2. Rows 4,5 of U_322.
    # B is 2x2. Rows 0-3 of V_322.
    # C_bot is 1x2. Rows 4,5 of W_322.
    
    U_sc = np.zeros((2, 4))
    V_sc = np.zeros((4, 4))
    W_sc = np.zeros((2, 4))
    
    # Same as U_122, V_122, W_122
    U_sc = U_122
    V_sc = V_122
    W_sc = W_122
    
    # Combine into U_322, V_322, W_322
    U_322 = np.zeros((6, 11))
    V_322 = np.zeros((4, 11))
    W_322 = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_322[0:4, 0:7] = U_s
    V_322[0:4, 0:7] = V_s
    W_322[0:4, 0:7] = W_s
    
    # Schoolbook part (cols 7-10)
    U_322[4:6, 7:11] = U_sc
    V_322[0:4, 7:11] = V_sc
    W_322[4:6, 7:11] = W_sc
    
    # Now construct M_{3,4,4} = M_{3,2,2} x M_{1,2,2}
    # U: 12x44
    # V: 16x44
    # W: 12x44
    
    U_final = np.zeros((12, 44))
    V_final = np.zeros((16, 44))
    W_final = np.zeros((12, 44))
    
    # Index mapping
    # U_322 row (i1, j1) -> i1 in {0,1,2}, j1 in {0,1}
    # U_122 row (i2, j2) -> i2 in {0}, j2 in {0,1}
    # Global U row (i, j) -> i in {0,1,2}, j in {0,1,2,3}
    # j = j1 * 2 + j2
    
    for i1 in range(3):
        for j1 in range(2):
            for i2 in range(1):
                for j2 in range(2):
                    j_global = j1 * 2 + j2
                    row_u = i1 * 4 + j_global
                    # U_final[row_u, col] = U_322[i1*2+j1, c1] * U_122[i2*2+j2, c2]
                    for c1 in range(11):
                        for c2 in range(4):
                            col = c1 * 4 + c2
                            U_final[row_u, col] = U_322[i1 * 2 + j1, c1] * U_122[i2 * 2 + j2, c2]
                    
    # V_322 row (j1, k1) -> j1 in {0,1}, k1 in {0,1}
    # V_122 row (j2, k2) -> j2 in {0,1}, k2 in {0,1}
    # Global V row (j, k) -> j in {0,1,2,3}, k in {0,1,2,3}
    # j = j1 * 2 + j2, k = k1 * 2 + k2
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j_global = j1 * 2 + j2
                    k_global = k1 * 2 + k2
                    row_v = j_global * 4 + k_global
                    for c1 in range(11):
                        for c2 in range(4):
                            col = c1 * 4 + c2
                            V_final[row_v, col] = V_322[j1 * 2 + k1, c1] * V_122[j2 * 2 + k2, c2]
                            
    # W_322 row (k1, i1) -> k1 in {0,1}, i1 in {0,1,2}
    # W_122 row (k2, i2) -> k2 in {0,1}, i2 in {0}
    # Global W row (k, i) -> k in {0,1,2,3}, i in {0,1,2}
    # k = k1 * 2 + k2
    
    for k1 in range(2):
        for i1 in range(3):
            for k2 in range(2):
                for i2 in range(1):
                    k_global = k1 * 2 + k2
                    row_w = k_global * 3 + i1
                    for c1 in range(11):
                        for c2 in range(4):
                            col = c1 * 4 + c2
                            W_final[row_w, col] = W_322[k1 * 3 + i1, c1] * W_122[k2 * 1 + i2, c2]
                            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
