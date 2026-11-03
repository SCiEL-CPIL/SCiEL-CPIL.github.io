import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices: 0:00, 1:01, 2:10, 3:11
    # r=0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # r=1: M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1 # C10, C11? No, C10, C11?
    # M2 = (a10+a11)b00
    # C10 += M2, C11 += -M2
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # r=2: M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    # M3 = a00(b01-b11)
    # C01 += M3, C11 += M3
    W2[1, 2] = 1
    W2[3, 2] = 1
    
    # r=3: M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    # M4 = a11(b10-b00)
    # C00 += M4, C10 += M4
    W2[0, 3] = 1
    W2[2, 3] = 1
    
    # r=4: M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    # M5 = (a00+a01)b11
    # C00 -= M5, C01 += M5
    W2[0, 4] = -1
    W2[1, 4] = 1
    
    # r=5: M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    # M6 = (a10-a00)(b00+b01)
    # C11 += M6
    W2[3, 5] = 1
    
    # r=6: M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1 # C00, C11
    # M7 = (a01-a11)(b10+b11)
    # C00 += M7, C11 += M7
    W2[0, 6] = 1
    W2[3, 6] = 1
    
    # Construct 4x4 factors
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r_block in range(7):
        for r_inner in range(7):
            r = r_block * 7 + r_inner
            for i in range(4):
                for j in range(4):
                    I, i_in = divmod(i, 2)
                    J, j_in = divmod(j, 2)
                    idx_u = i * 4 + j
                    val_u = U2[i_in * 2 + j_in, r_inner] * U2[I * 2 + J, r_block]
                    U4[idx_u, r] = val_u
                    
                    idx_v = j * 4 + k # Wait, V index is j*p+k. Here p=4.
                    # V4 index is j*4 + k.
                    # j is row of B, k is col of B.
                    # In 4x4, j, k in 0..3.
                    pass # Need to fill V4 and W4 similarly
            
            for j in range(4):
                for k in range(4):
                    J, j_in = divmod(j, 2)
                    K, k_in = divmod(k, 2)
                    idx_v = j * 4 + k
                    val_v = V2[j_in * 2 + k_in, r_inner] * V2[J * 2 + K, r_block]
                    V4[idx_v, r] = val_v
            
            for k in range(4):
                for i in range(4):
                    K, k_in = divmod(k, 2)
                    I, i_in = divmod(i, 2)
                    idx_w = k * 4 + i
                    val_w = W2[k_in * 2 + i_in, r_inner] * W2[K * 2 + I, r_block]
                    W4[idx_w, r] = val_w

    # Construct 4x4x8 factors
    # Rank 98
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    V[0:16, 0:49] = V4
    V[16:32, 49:98] = V4
    
    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
