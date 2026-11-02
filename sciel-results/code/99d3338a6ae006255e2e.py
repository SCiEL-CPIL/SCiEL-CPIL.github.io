import numpy as np

def construct():
    """
    Returns a rank-39 decomposition for 3x3x6 matrix multiplication.
    U: (9, 39), V: (18, 39), W: (18, 39)
    """
    n, m, p = 3, 3, 6
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 3x3x2 block decomposition (Rank 13)
    # U_block: 9x13, V_block: 6x13, W_block: 6x13
    U_block = np.zeros((9, 13), dtype=float)
    V_block = np.zeros((6, 13), dtype=float)
    W_block = np.zeros((6, 13), dtype=float)
    
    # Strassen 2x2x2 (7 terms) for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Indices in block: A[i,j] -> i*3+j, B[j,k] -> j*2+k, C[k,i] -> k*3+i
    
    # M1 = (A00+A11)*(B00+B11)
    U_block[0,0] = 1; U_block[4,0] = 1
    V_block[0,0] = 1; V_block[3,0] = 1
    W_block[0,0] = 1; W_block[4,0] = 1
    
    # M2 = (A10+A11)*B00
    U_block[3,1] = 1; U_block[4,1] = 1
    V_block[0,1] = 1
    W_block[1,1] = 1
    
    # M3 = A00*(B01-B11)
    U_block[0,2] = 1
    V_block[1,2] = 1; V_block[3,2] = -1
    W_block[3,2] = 1
    
    # M4 = A11*(B10-B00)
    U_block[4,3] = 1
    V_block[2,3] = 1; V_block[0,3] = -1
    W_block[0,3] = -1; W_block[1,3] = 1
    
    # M5 = (A00+A01)*B11
    U_block[0,4] = 1; U_block[1,4] = 1
    V_block[3,4] = 1
    W_block[0,4] = 1; W_block[3,4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U_block[3,5] = 1; U_block[0,5] = -1
    V_block[0,5] = 1; V_block[1,5] = 1
    W_block[4,5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U_block[1,6] = 1; U_block[4,6] = -1
    V_block[2,6] = 1; V_block[3,6] = 1
    W_block[0,6] = -1; W_block[4,6] = -1
    
    # Cross terms (6 terms)
    # t1 = (A02+A12)*B20
    U_block[2,7] = 1; U_block[5,7] = 1
    V_block[4,7] = 1
    W_block[0,7] = 1; W_block[1,7] = 1
    
    # t2 = (A02+A12)*B21
    U_block[2,8] = 1; U_block[5,8] = 1
    V_block[5,8] = 1
    W_block[3,8] = 1; W_block[4,8] = 1
    
    # t3 = A20*(B00+B01)
    U_block[6,9] = 1
    V_block[0,9] = 1; V_block[1,9] = 1
    W_block[2,9] = 1; W_block[5,9] = 1
    
    # t4 = A21*(B10+B11)
    U_block[7,10] = 1
    V_block[2,10] = 1; V_block[3,10] = 1
    W_block[2,10] = 1; W_block[5,10] = 1
    
    # t5 = A22*B20
    U_block[8,11] = 1
    V_block[4,11] = 1
    W_block[2,11] = 1
    
    # t6 = A22*B21
    U_block[8,12] = 1
    V_block[5,12] = 1
    W_block[5,12] = 1
    
    # Assemble 3x3x6 by replicating the block 3 times
    for g in range(3):
        start = g * 13
        end = (g + 1) * 13
        
        # U is the same for all blocks
        U[:, start:end] = U_block
        
        # V: block g handles B columns 2g, 2g+1
        # V_block index j*2+k maps to V index j*6 + 2g + k
        for j in range(3):
            for k in range(2):
                V[j * 6 + 2 * g + k, start:end] = V_block[j * 2 + k, :]
        
        # W: block g handles C columns 2g, 2g+1
        # W_block index k*3+i maps to W index (2g+k)*3 + i
        for k in range(2):
            for i in range(3):
                W[(2 * g + k) * 3 + i, start:end] = W_block[k * 3 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
