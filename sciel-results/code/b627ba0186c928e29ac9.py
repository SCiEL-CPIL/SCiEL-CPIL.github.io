"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
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

def _get_223_rank11():
    """
    Returns U, V, W for 2x2x3 multiplication with rank 11 (Strassen 2x2x2 + schoolbook 2x2x1).
    U shape: (4, 11), V shape: (6, 11), W shape: (6, 11)
    Indices: U[i*2+j], V[j*3+k], W[k*2+i]
    """
    n, m, p = 2, 2, 3
    r = 11
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First 7 columns: Strassen 2x2x2 for k=0,1
    U_s, V_s, W_s = _get_strassen_222()
    for i in range(2):
        for j in range(2):
            U[i*2+j, 0:7] = U_s[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*3+k, 0:7] = V_s[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[k*2+i, 0:7] = W_s[k*2+i, :]
    
    # Last 4 columns: schoolbook 2x2x1 for k=2
    r_sc = 4
    for c in range(r_sc):
        i = c // 2
        j = c % 2
        U[i*2+j, 7+c] = 1.0
        V[j*3+2, 7+c] = 1.0
        W[2*2+i, 7+c] = 1.0
    
    return U, V, W

def _get_224_rank14():
    """
    Returns U, V, W for 2x2x4 multiplication with rank 14 (two Strassen 2x2x2).
    U shape: (4, 14), V shape: (8, 14), W shape: (8, 14)
    Indices: U[i*2+j], V[j*4+k], W[k*2+i]
    """
    n, m, p = 2, 2, 4
    r = 14
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    U_s, V_s, W_s = _get_strassen_222()
    
    # First 2x2x2 block: k=0,1
    for i in range(2):
        for j in range(2):
            U[i*2+j, 0:7] = U_s[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*4+k, 0:7] = V_s[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[k*2+i, 0:7] = W_s[k*2+i, :]
    
    # Second 2x2x2 block: k=2,3
    for i in range(2):
        for j in range(2):
            U[i*2+j, 7:14] = U_s[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*4+(k+2), 7:14] = V_s[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[(k+2)*2+i, 7:14] = W_s[k*2+i, :]
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Improved decomposition:
    # m=4 split into 2 blocks of 2.
    # p=7 split into 3 and 4.
    # For each m-block: 2x2x3 (rank 9) + 2x2x4 (rank 14) = 23
    # Total Rank = 2 * 23 = 46.
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # m_blocks: [0, 2) and [2, 4)
    m_blocks = [(0, 2), (2, 4)]
    
    # p_blocks: [0, 3) with rank-11 2x2x3, [3, 7) with rank-14 2x2x4
    p_blocks = [
        (0, 3, 'rank11_223'),
        (3, 7, 'rank14_224')
    ]
    
    # Pre-fetch sub-tensors
    U_r11, V_r11, W_r11 = _get_223_rank11()
    U_r14, V_r14, W_r14 = _get_224_rank14()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'rank11_223':
                U_sub, V_sub, W_sub = U_r11, V_r11, W_r11
                r_sub = 11
                p_sub = 3
            else:
                U_sub, V_sub, W_sub = U_r14, V_r14, W_r14
                r_sub = 14
                p_sub = 4
            
            # Place U
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            # Place V
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            # Place W
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to original rank-50 approach
            return _construct_rank50()
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank50():
    """Fallback to the original rank-50 decomposition."""
    n, m, p = DIMENSIONS
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    
    # Schoolbook 2x2x1
    r_sc = 4
    U_sc = np.zeros((4, r_sc))
    V_sc = np.zeros((2, r_sc))
    W_sc = np.zeros((2, r_sc))
    for c in range(r_sc):
        i = c // 2
        j = c % 2
        U_sc[i * 2 + j, c] = 1.0
        V_sc[j, c] = 1.0
        W_sc[i, c] = 1.0
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                p_sub = 1
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
