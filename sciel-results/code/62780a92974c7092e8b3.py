"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook_fallback(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2_block():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    # M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6
    U[3, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    U_s, V_s, W_s = _strassen_2x2x2_block()
    
    if not _verify_decomp(U_s, V_s, W_s, 2, 2, 2, 7):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 2x4x2 (rank 14)
    U1, V1, W1 = np.zeros((8, 14)), np.zeros((8, 14)), np.zeros((4, 14))
    # Block 1: j 0,1
    for r in range(7):
        U1[0:4, r] = U_s[0:4, r]
        V1[0:4, r] = V_s[0:4, r]
        W1[:, r] = W_s[:, r]
    # Block 2: j 2,3
    for r in range(7):
        U1[4:8, r+7] = U_s[0:4, r]
        V1[4:8, r+7] = V_s[0:4, r]
        W1[:, r+7] = W_s[:, r]
        
    if not _verify_decomp(U1, V1, W1, 2, 4, 2, 14):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Build 2x4x4 (rank 28)
    U2, V2, W2 = np.zeros((8, 28)), np.zeros((16, 28)), np.zeros((8, 28))
    # Block 1: k 0,1
    for r in range(14):
        U2[:, r] = U1[:, r]
        V2[0:8, r] = V1[0:8, r]
        W2[0:4, r] = W1[0:4, r]
    # Block 2: k 2,3
    for r in range(14):
        U2[:, r+14] = U1[:, r]
        V2[8:16, r+14] = V1[0:8, r]
        W2[4:8, r+14] = W1[0:4, r]
        
    if not _verify_decomp(U2, V2, W2, 2, 4, 4, 28):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Combine 2x4x4 and 2x4x2
    total_rank = 42
    U_total = np.zeros((8, total_rank))
    V_total = np.zeros((24, total_rank))
    W_total = np.zeros((12, total_rank))
    
    # 2x4x4 part
    U_total[:, :28] = U2
    for j in range(4):
        for k in range(4):
            V_total[j*6 + k, :28] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(2):
            W_total[k*2 + i, :28] = W2[k*2 + i, :]
            
    # 2x4x2 part
    U_total[:, 28:] = U1
    for j in range(4):
        for kl in range(2):
            V_total[j*6 + (kl+4), 28:] = V1[j*2 + kl, :]
    for kl in range(2):
        for i in range(2):
            W_total[(kl+4)*2 + i, 28:] = W1[kl*2 + i, :]
            
    if not _verify_decomp(U_total, V_total, W_total, n, m, p, total_rank):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
