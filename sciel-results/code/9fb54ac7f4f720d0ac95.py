import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Base case: 2x2x2 Strassen decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # r=0: M1 = A00 * (B00 + B01)
    # L: A00=1. K: B00=1, B01=1. W: C00=1, C11=1.
    U2[0, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C[0,0] idx 0, C[1,1] idx 3
    
    # r=1: M2 = (A00 + A01) * B11
    # L: A00=1, A01=1. K: B11=1. W: C01=1.
    U2[0, 1] = 1; U2[1, 1] = 1
    V2[3, 1] = 1
    W2[1, 1] = 1 # C[0,1] idx 1
    
    # r=2: M3 = (A10 + A11) * B00
    # L: A10=1, A11=1. K: B00=1. W: C10=1.
    U2[2, 2] = 1; U2[3, 2] = 1
    V2[0, 2] = 1
    W2[2, 2] = 1 # C[1,0] idx 2
    
    # r=3: M4 = A11 * (B10 + B11)
    # L: A11=1. K: B10=1, B11=1. W: C00=-1, C01=1.
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[3, 3] = 1
    W2[0, 3] = -1; W2[1, 3] = 1
    
    # r=4: M5 = (A00 + A10) * (B01 + B11)
    # L: A00=1, A10=1. K: B01=1, B11=1. W: C10=1.
    U2[0, 4] = 1; U2[2, 4] = 1
    V2[1, 4] = 1; V2[3, 4] = 1
    W2[2, 4] = 1
    
    # r=5: M6 = (A01 - A11) * (B10 + B11)
    # L: A01=1, A11=-1. K: B10=1, B11=1. W: C11=1.
    U2[1, 5] = 1; U2[3, 5] = -1
    V2[2, 5] = 1; V2[3, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A00 - A01) * (B00 + B10)
    # L: A00=1, A01=-1. K: B00=1, B10=1. W: C00=1.
    U2[0, 6] = 1; U2[1, 6] = -1
    V2[0, 6] = 1; V2[2, 6] = 1
    W2[0, 6] = 1
    
    # Now build 4x4x4 decomposition
    # Rank 49
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Strassen coefficients for 4x4x4 (2x2 blocks of 2x2)
    # I, J, K in {0, 1}
    # L_r[I, J], K_r[J, K], W_r[K, I]
    
    Ls = []
    Ks = []
    Ws = []
    
    # r=0: M1 = A00 * (B00 + B01)
    Ls.append(np.array([[1, 0], [0, 0]]))
    Ks.append(np.array([[1, 1], [0, 0]]))
    Ws.append(np.array([[1, 0], [0, 1]])) # C00, C11
    
    # r=1: M2 = (A00 + A01) * B11
    Ls.append(np.array([[1, 1], [0, 0]]))
    Ks.append(np.array([[0, 0], [0, 1]]))
    Ws.append(np.array([[0, 1], [0, 0]])) # C01
    
    # r=2: M3 = (A10 + A11) * B00
    Ls.append(np.array([[0, 0], [1, 1]]))
    Ks.append(np.array([[1, 0], [0, 0]]))
    Ws.append(np.array([[0, 0], [1, 0]])) # C10
    
    # r=3: M4 = A11 * (B10 + B11)
    Ls.append(np.array([[0, 0], [0, 1]]))
    Ks.append(np.array([[0, 0], [1, 1]]))
    Ws.append(np.array([[-1, 1], [0, 0]])) # C00, C01
    
    # r=4: M5 = (A00 + A10) * (B01 + B11)
    Ls.append(np.array([[1, 0], [1, 0]]))
    Ks.append(np.array([[0, 1], [0, 1]]))
    Ws.append(np.array([[0, 0], [1, 0]])) # C10
    
    # r=5: M6 = (A01 - A11) * (B10 + B11)
    Ls.append(np.array([[0, 1], [0, -1]]))
    Ks.append(np.array([[0, 0], [1, 1]]))
    Ws.append(np.array([[0, 0], [0, 1]])) # C11
    
    # r=6: M7 = (A00 - A01) * (B00 + B10)
    Ls.append(np.array([[1, -1], [0, 0]]))
    Ks.append(np.array([[1, 0], [1, 0]]))
    Ws.append(np.array([[1, 0], [0, 0]])) # C00
    
    col = 0
    for r in range(7):
        L = Ls[r]
        K = Ks[r]
        W = Ws[r]
        for s in range(7):
            # For each base term s, and block indices I, J, K
            for I in range(2):
                for J in range(2):
                    for K_idx in range(2): # K is used in W, J in K, I in L
                        l_val = L[I, J]
                        k_val = K[J, K_idx]
                        w_val = W[K_idx, I]
                        
                        if l_val == 0 and k_val == 0 and w_val == 0:
                            continue
                            
                        # Iterate over local indices i, j, k in 2x2 blocks
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    # U index: (I*2+i)*4 + (J*2+j)
                                    u_idx = (I*2 + i)*4 + (J*2 + j)
                                    # V index: (J*2+j)*4 + (K_idx*2+k)
                                    v_idx = (J*2 + j)*4 + (K_idx*2 + k)
                                    # W index: (K_idx*2+k)*4 + (I*2+i)
                                    w_idx = (K_idx*2 + k)*4 + (I*2 + i)
                                    
                                    # Base indices
                                    u_base = i*2 + j
                                    v_base = j*2 + k
                                    w_base = k*2 + i
                                    
                                    u_val = l_val * U2[u_base, s]
                                    v_val = k_val * V2[v_base, s]
                                    w_val_final = w_val * W2[w_base, s]
                                    
                                    if u_val != 0:
                                        U4[u_idx, col] = u_val
                                    if v_val != 0:
                                        V4[v_idx, col] = v_val
                                    if w_val_final != 0:
                                        W4[w_idx, col] = w_val_final
            col += 1
            
    # Now build 4x4x8 decomposition
    # Rank 98
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # First half (B1)
    U[:, 0:49] = U4
    # V4 has shape 16x49. Rows are j*4+k (k in 0..3).
    # Full V rows are j*8+k.
    # So row r in V4 maps to row r in V? No.
    # V4 row index: j*4+k. j in 0..3, k in 0..3.
    # Full V row index: j*8+k.
    # Since k < 4, j*4+k != j*8+k.
    # We need to map V4 rows to V rows.
    for j in range(4):
        for k in range(4):
            v4_row = j*4 + k
            v_full_row = j*8 + k
            V[v_full_row, 0:49] = V4[v4_row, 0:49]
            
    # W4 has shape 16x49. Rows are k*4+i (k in 0..3).
    # Full W rows are k*4+i.
    # So row r in W4 maps to row r in W.
    W[0:16, 0:49] = W4
    
    # Second half (B2)
    U[:, 49:98] = U4
    
    # V4 rows map to V rows for B2 (k in 4..7)
    for j in range(4):
        for k in range(4):
            v4_row = j*4 + k
            v_full_row = j*8 + (k+4)
            V[v_full_row, 49:98] = V4[v4_row, 0:49]
            
    # W4 rows map to W rows for C2 (k in 4..7)
    for k in range(4):
        for i in range(4):
            w4_row = k*4 + i
            w_full_row = (k+4)*4 + i
            W[w_full_row, 49:98] = W4[w4_row, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
