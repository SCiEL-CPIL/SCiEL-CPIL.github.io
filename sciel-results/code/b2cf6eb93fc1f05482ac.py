import numpy as np

def construct():
    n, m, p = 5, 5, 5
    # Strassen 2x2 decomposition
    # U, V, W are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = 1 # W index (k,i). C10 is i=1,k=0 -> (0,1). C11 is i=1,k=1 -> (1,1).
    # Wait, W2 row index is k*n + i. n=2.
    # C10: i=1, k=0. Index 0*2 + 1 = 1.
    # C11: i=1, k=1. Index 1*2 + 1 = 3.
    # So W2[1, 1] and W2[3, 1]. Correct.
    
    # M3: A00(B10-B11) -> C01, C11
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01: i=0,k=1 -> 1*2+0=2. C11: i=1,k=1 -> 3.
    
    # M4: A11(B01-B00) -> C00, C10
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00: i=0,k=0 -> 0. C10: i=1,k=0 -> 1.
    
    # M5: (A00+A01)B11 -> C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00: -1. C01: 1.
    
    # M6: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11.
    
    # M7: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00.

    # 4x4 Strassen decomposition
    # U4 = U2 x U2 (Kronecker)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 5x5 decomposition
    # Initialize with zeros
    rank_5 = 54
    U5 = np.zeros((25, rank_5), dtype=int)
    V5 = np.zeros((25, rank_5), dtype=int)
    W5 = np.zeros((25, rank_5), dtype=int)
    
    # Embed 4x4 part (indices 0..3)
    # U4 is 16x49. Rows are i*4+j for i,j in 0..3.
    # U5 rows are i*5+j for i,j in 0..4.
    for i in range(4):
        for j in range(4):
            row_4 = i * 4 + j
            row_5 = i * 5 + j
            U5[row_5, :49] = U4[row_4, :]
            
    for j in range(4):
        for k in range(4):
            row_4 = j * 4 + k
            row_5 = j * 5 + k
            V5[row_5, :49] = V4[row_4, :]
            
    for k in range(4):
        for i in range(4):
            row_4 = k * 4 + i
            row_5 = k * 5 + i
            W5[row_5, :49] = W4[row_4, :]
            
    # Add 5 cross terms
    # 1. i<4, j<4, k=4: A_i4 B_4k? No, A_ij B_j4. j<4.
    # Term: A_i,j B_j,4. i<4, j<4. Contributes to C_i,4.
    # U_(i,j), V_(j,4), W_(4,i).
    # U5[i*5+j, 49] = 1
    # V5[j*5+4, 49] = 1
    # W5[4*5+i, 49] = 1
    for i in range(4):
        for j in range(4):
            U5[i*5+j, 49] = 1
            V5[j*5+4, 49] = 1
            W5[4*5+i, 49] = 1
            
    # 2. i<4, j=4, k<4: A_i4 B_4k.
    # U5[i*5+4, 50] = 1
    # V5[4*5+k, 50] = 1
    # W5[k*5+i, 50] = 1
    for i in range(4):
        for k in range(4):
            U5[i*5+4, 50] = 1
            V5[4*5+k, 50] = 1
            W5[k*5+i, 50] = 1
            
    # 3. i<4, j=4, k=4: A_i4 B_44.
    # U5[i*5+4, 51] = 1
    # V5[4*5+4, 51] = 1
    # W5[4*5+i, 51] = 1
    for i in range(4):
        U5[i*5+4, 51] = 1
        V5[4*5+4, 51] = 1
        W5[4*5+i, 51] = 1
        
    # 4. i=4, j<4, k<4: A_4j B_jk.
    # U5[4*5+j, 52] = 1
    # V5[j*5+k, 52] = 1
    # W5[k*5+4, 52] = 1
    for j in range(4):
        for k in range(4):
            U5[4*5+j, 52] = 1
            V5[j*5+k, 52] = 1
            W5[k*5+4, 52] = 1
            
    # 5. i=4, j=4, k=4: A_44 B_44.
    U5[4*5+4, 53] = 1
    V5[4*5+4, 53] = 1
    W5[4*5+4, 53] = 1

    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
