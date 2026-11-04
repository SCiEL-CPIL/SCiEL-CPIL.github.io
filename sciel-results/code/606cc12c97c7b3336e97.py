import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2 factors
    # Rank 7
    # Indices for 2x2: i,j in {0,1}
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Using standard Strassen coefficients
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # We need U, V, W such that T = sum U_r V_r W_r^T (conceptually)
    # The seed uses 1s. Strassen uses +/- 1.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Helper to set indices
    # i,j for U (A), j,k for V (B), k,i for W (C)
    # 0: 11, 1: 12, 2: 21, 3: 22 (row-major)
    
    # P1: A11+A22, B11+B22, C11+C22 (coeff 1)
    # A11(0), A22(3) -> U_s[0,0]=1, U_s[3,0]=1
    # B11(0), B22(3) -> V_s[0,0]=1, V_s[3,0]=1
    # C11(0), C22(3) -> W_s[0,0]=1, W_s[3,0]=1
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[3,0] = 1
    
    # P2: A21+A22, B11, C21
    # A21(2), A22(3)
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1 # C21 is index 2 (row 1, col 0? No, W is k*n+i. C21 is k=1, i=0? No. C is 2x2. C_{21} is row 1 col 0. Index 1*2+0=2)
    
    # P3: A11, B12-B22, C12
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1 # C12 is index 1
    
    # P4: A22, B21-B11, C21
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[2,3] = 1
    
    # P5: A11+A12, B22, C11
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = 1
    
    # P6: A21-A11, B11+B12, C22
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1
    
    # P7: A12-A22, B21+B22, C11
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = 1
    
    # 2. Schoolbook 3x3 factors
    # Rank 27
    U_3 = np.zeros((9, 27))
    V_3 = np.zeros((9, 27))
    W_3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_3[i*3+j, col] = 1
                V_3[j*3+k, col] = 1
                W_3[k*3+i, col] = 1
                col += 1
                
    # 3. Kronecker Product
    U_kron = np.kron(U_s, U_3)
    V_kron = np.kron(V_s, V_3)
    W_kron = np.kron(W_s, W_3)
    
    # Shapes: (36, 189)
    
    # 4. Permutation
    # Map from (i1, j1, i2, j2) to (i1, i2, j1, j2)
    # n1=2, n2=3
    # old_idx = i1*9 + j1*9 + i2*3 + j2 ? No.
    # U_s index: i1*2 + j1. Size 4.
    # U_3 index: i2*3 + j2. Size 9.
    # Kron index: (i1*2 + j1)*9 + (i2*3 + j2)
    
    # New index for 6x6:
    # I = i1*3 + i2
    # J = j1*3 + j2
    # new_idx = I*6 + J = (i1*3 + i2)*6 + (j1*3 + j2)
    
    perm = np.zeros(36, dtype=int)
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(3):
                for j2 in range(3):
                    old_idx = (i1*2 + j1)*9 + (i2*3 + j2)
                    new_idx = (i1*3 + i2)*6 + (j1*3 + j2)
                    perm[new_idx] = old_idx
                    
    U_new = U_kron[perm]
    V_new = V_kron[perm]
    W_new = W_kron[perm]
    
    # Convert to lists of lists as required
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
