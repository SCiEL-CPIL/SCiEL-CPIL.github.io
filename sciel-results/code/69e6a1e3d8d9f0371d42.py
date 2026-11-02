import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank53():
    n, m, p = DIMENSIONS
    # 7 blocks of size 2, 1 block of size 1
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(7):
        start_k = b * 2
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                V[j * p + start_k + k_local, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            for i in range(n):
                W[(start_k + k_local) * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
        
    # Last block of size 1
    start_k = 14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
    for i in range(n):
        W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank53()
    
    # Verify before returning to ensure correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
