import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-7 Strassen decomposition for 2x2 matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0:11, 1:12, 2:21, 3:22
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a11
        [0, 0, 0, 0, 1, 0, 1],  # a12
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [1, 1, 0, 1, 0, 0, -1]  # a22
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b11
        [0, 0, 1, 0, 0, 1, 0],  # b12
        [0, 0, 0, 1, 0, 0, 1],  # b21
        [1, 0, -1, 0, 1, 0, 1]  # b22
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c11
        [0, 0, 1, 0, 1, 0, 0],  # c12
        [0, 1, 0, 1, 0, 0, 0],  # c21
        [1, -1, 1, 0, 0, 1, 0]  # c22
    ])
    
    # Check 2x2 decomposition validity
    # C = W.T @ (U * V.T) ? No.
    # C[i, k] = sum_c W[k, c] U[i, c] V[j, c] ? No.
    # Tensor T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    # For 2x2:
    # U is 4x7, V is 4x7, W is 4x7.
    # i, j, k in 0..3.
    # i maps to A row/col, j to B row/col, k to C row/col.
    
    # Kronecker product
    U_raw = np.kron(U2, U2)
    V_raw = np.kron(V2, V2)
    W_raw = np.kron(W2, W2)
    
    # Permutations
    # U: raw index u_raw = u1 * 4 + u2
    # u1 = 2*i1 + j1, u2 = 2*i2 + j2
    # target index r = 4*i + j = 4*(2*i1 + i2) + (2*j1 + j2) = 8*i1 + 4*i2 + 2*j1 + j2
    perm_U = [0] * 16
    for u1 in range(4):
        i1 = u1 // 2
        j1 = u1 % 2
        for u2 in range(4):
            i2 = u2 // 2
            j2 = u2 % 2
            u_raw = u1 * 4 + u2
            r = 8 * i1 + 4 * i2 + 2 * j1 + j2
            perm_U[r] = u_raw
            
    # V: raw index v_raw = v1 * 4 + v2
    # v1 = 2*j1 + k1, v2 = 2*j2 + k2
    # target index b = 4*j + k = 4*(2*j1 + j2) + (2*k1 + k2) = 8*j1 + 4*j2 + 2*k1 + k2
    perm_V = [0] * 16
    for v1 in range(4):
        j1 = v1 // 2
        k1 = v1 % 2
        for v2 in range(4):
            j2 = v2 // 2
            k2 = v2 % 2
            v_raw = v1 * 4 + v2
            b = 8 * j1 + 4 * j2 + 2 * k1 + k2
            perm_V[b] = v_raw
            
    # W: raw index w_raw = w1 * 4 + w2
    # w1 = 2*k1 + i1, w2 = 2*k2 + i2
    # target index c = 4*k + i = 4*(2*k1 + k2) + (2*i1 + i2) = 8*k1 + 4*k2 + 2*i1 + i2
    perm_W = [0] * 16
    for w1 in range(4):
        k1 = w1 // 2
        i1 = w1 % 2
        for w2 in range(4):
            k2 = w2 // 2
            i2 = w2 % 2
            w_raw = w1 * 4 + w2
            c = 8 * k1 + 4 * k2 + 2 * i1 + i2
            perm_W[c] = w_raw
            
    U_final = U_raw[perm_U]
    V_final = V_raw[perm_V]
    W_final = W_raw[perm_W]
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
