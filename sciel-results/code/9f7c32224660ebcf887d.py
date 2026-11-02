import time
import numpy as np

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _embed_block(U_sub, V_sub, W_sub, n, m, p, k_start):
    p_sub = W_sub.shape[0] // n  # or len(W_sub[0]) if list, but we use numpy here
    # Actually W_sub is from _schoolbook_decomp which returns lists.
    # Let's convert to numpy for embedding.
    U_sub = np.array(U_sub)
    V_sub = np.array(V_sub)
    W_sub = np.array(W_sub)
    
    p_sub = W_sub.shape[0] // n
    
    U_full = np.zeros((n*m, U_sub.shape[1]))
    U_full[:] = U_sub
    
    V_full = np.zeros((m*p, V_sub.shape[1]))
    for j in range(m):
        for k_local in range(p_sub):
            k_global = k_start + k_local
            V_full[j*p + k_global, :] = V_sub[j*p_sub + k_local, :]
            
    W_full = np.zeros((p*n, W_sub.shape[1]))
    for k_local in range(p_sub):
        k_global = k_start + k_local
        for i in range(n):
            W_full[k_global*n + i, :] = W_sub[k_local*n + i, :]
            
    return U_full, V_full, W_full

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    
    blocks = []
    
    # Block 1: k=0,1
    U1, V1, W1 = _schoolbook_decomp(n, m, 2)
    U1, V1, W1 = _embed_block(U1, V1, W1, n, m, p, 0)
    blocks.append((U1, V1, W1))
    
    # Block 2: k=2,3
    U2, V2, W2 = _schoolbook_decomp(n, m, 2)
    U2, V2, W2 = _embed_block(U2, V2, W2, n, m, p, 2)
    blocks.append((U2, V2, W2))
    
    # Block 3: k=4
    U3, V3, W3 = _schoolbook_decomp(n, m, 1)
    U3, V3, W3 = _embed_block(U3, V3, W3, n, m, p, 4)
    blocks.append((U3, V3, W3))
    
    U_full = np.hstack([b[0] for b in blocks])
    V_full = np.hstack([b[1] for b in blocks])
    W_full = np.hstack([b[2] for b in blocks])
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
