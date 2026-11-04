import numpy as np
import time

def get_strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using trivial (schoolbook) decomposition.
    U: (16, 64), V: (16, 64), W: (16, 64)
    Indices:
    u = i*4 + j
    v = j*4 + k
    w = k*4 + i
    """
    U4 = np.zeros((16, 64))
    V4 = np.zeros((16, 64))
    W4 = np.zeros((16, 64))
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r = i * 16 + j * 4 + k
                u = i * 4 + j
                v = j * 4 + k
                w = k * 4 + i
                U4[u, r] = 1
                V4[v, r] = 1
                W4[w, r] = 1
    
    return U4, V4, W4

def construct():
    n, m, p = 4, 4, 9
    
    # 1. Get 4x4x4 decomposition
    U_str, V_str, W_str = get_strassen_4x4()
    
    # 2. Construct 4x4x9 decomposition
    # Rank = 64 + 64 + 16 = 144
    
    rank = 144
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: B cols 0-3, C cols 0-3
    # U: Copy U_str
    U[:, 0:64] = U_str
    
    # V: Map V_str (16x64) to V (36x64)
    # V_str index v' = j*4 + k (j in 0..3, k in 0..3)
    # V index v = j*9 + k
    for v_prime in range(16):
        j = v_prime // 4
        k = v_prime % 4
        v = j * 9 + k
        V[v, 0:64] = V_str[v_prime, :]
        
    # W: Map W_str (16x64) to W (36x64)
    # W_str index w' = k*4 + i (k in 0..3, i in 0..3)
    # W index w = k*n + i = k*4 + i.
    # In Block 1, k in 0..3.
    # So w = k*4 + i. This is the same as w'.
    W[0:16, 0:64] = W_str
    
    # Block 2: B cols 4-7, C cols 4-7
    # U: Copy U_str
    U[:, 64:128] = U_str
    
    # V: Map V_str to V
    # B cols 4-7. k in 4..7.
    # v' = j*4 + k_local (k_local in 0..3)
    # k = 4 + k_local
    # v = j*9 + k = j*9 + 4 + k_local
    for v_prime in range(16):
        j = v_prime // 4
        k_local = v_prime % 4
        k = 4 + k_local
        v = j * 9 + k
        V[v, 64:128] = V_str[v_prime, :]
        
    # W: Map W_str to W
    # C cols 4-7. k in 4..7.
    # w' = k_local*4 + i
    # w = k*4 + i = (4+k_local)*4 + i = 16 + k_local*4 + i
    for w_prime in range(16):
        k_local = w_prime // 4
        i = w_prime % 4
        w = (4 + k_local) * 4 + i
        W[w, 64:128] = W_str[w_prime, :]
        
    # Block 3: B col 8, C col 8
    # Rank 16. Standard dot product.
    # C_{i,8} = sum_j A_{i,j} B_{j,8}
    # Term for (i,j):
    # u = i*4 + j
    # v = j*9 + 8
    # w = 8*4 + i = 32 + i
    
    for i in range(4):
        for j in range(4):
            k_idx = 128 + i*4 + j
            u = i * 4 + j
            v = j * 9 + 8
            w = 32 + i
            U[u, k_idx] = 1
            V[v, k_idx] = 1
            W[w, k_idx] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
