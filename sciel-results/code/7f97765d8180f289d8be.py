import numpy as np

def construct():
    # 1. Generate 2x2x2 Strassen factors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # r=4: M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # r=5: M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[1, 6] = -1
    
    # 2. Construct 4x4x4 factors (Rank 49)
    # U4, V4, W4 are 16x49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            i1, i2 = divmod(i, 2)
            j1, j2 = divmod(j, 2)
            idx_U = i * 4 + j
            idx_U1 = i1 * 2 + j1
            idx_U2 = i2 * 2 + j2
            
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U4[idx_U, r] = U2[idx_U1, r1] * U2[idx_U2, r2]
                    
                    # V4 depends on j, k
                    # We need to fill V4 for all j, k
                    # This loop is inefficient for V4 and W4. 
                    # Better to loop j, k for V4 and k, i for W4.
                    
    # Redo V4 and W4 construction properly
    V4 = np.zeros((16, 49), dtype=int)
    for j in range(4):
        for k in range(4):
            j1, j2 = divmod(j, 2)
            k1, k2 = divmod(k, 2)
            idx_V = j * 4 + k
            idx_V1 = j1 * 2 + k1
            idx_V2 = j2 * 2 + k2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V4[idx_V, r] = V2[idx_V1, r1] * V2[idx_V2, r2]
                    
    W4 = np.zeros((16, 49), dtype=int)
    for k in range(4):
        for i in range(4):
            k1, k2 = divmod(k, 2)
            i1, i2 = divmod(i, 2)
            idx_W = k * 4 + i
            idx_W1 = k1 * 2 + i1
            idx_W2 = k2 * 2 + i2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W4[idx_W, r] = W2[idx_W1, r1] * W2[idx_W2, r2]

    # 3. Construct 4x4x8 factors (Rank 98)
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # U is same for both halves
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    # V
    for j in range(m):
        for k in range(p):
            idx_V_full = j * p + k
            if k < 4:
                idx_V_sub = j * 4 + k
                V[idx_V_full, 0:49] = V4[idx_V_sub, :]
            else:
                k_sub = k - 4
                idx_V_sub = j * 4 + k_sub
                V[idx_V_full, 49:98] = V4[idx_V_sub, :]
                
    # W
    for k in range(p):
        for i in range(n):
            idx_W_full = k * n + i
            if k < 4:
                idx_W_sub = k * 4 + i
                W[idx_W_full, 0:49] = W4[idx_W_sub, :]
            else:
                k_sub = k - 4
                idx_W_sub = k_sub * 4 + i
                W[idx_W_full, 49:98] = W4[idx_W_sub, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
