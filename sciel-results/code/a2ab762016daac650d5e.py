"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication (Strassen, rank 7)."""
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

def _get_school_221():
    """Returns U, V, W for 2x2x1 multiplication (schoolbook, rank 4)."""
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    return U, V, W

def _get_school_222():
    """Returns U, V, W for 2x2x2 multiplication (schoolbook, rank 4)."""
    n, m, p = 2, 2, 2
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c in range(r):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify tensor identity: sum_c W[k*n+i,c]*U[i*m+j,c]*V[jp*p+k,c] == delta_{j,jp}"""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = np.sum(W[k * n + i, :] * U[i * m + j, :] * V[jp * p + k, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_decomp(n, m, p, m_split, p_split, deadline=None):
    """
    Build block decomposition.
    m_split: list of (start, end) for m dimension blocks
    p_split: list of (start, end, type) for p dimension blocks, type in {'strassen', 'school'}
    """
    if deadline and time.time() > deadline:
        return None
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    U_sb, V_sb, W_sb = _get_school_222()
    
    total_rank = 0
    for j_start, j_end in m_split:
        for k_start, k_end, p_type in p_split:
            if p_type == 'strassen':
                total_rank += 7
            elif p_type == 'school':
                total_rank += 4
            else:
                total_rank += 4
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    
    for j_start, j_end in m_split:
        m_sub = j_end - j_start
        for k_start, k_end, p_type in p_split:
            p_sub = k_end - k_start
            if p_type == 'strassen':
                U_sub, V_sub, W_sub, r_sub = U_s, V_s, W_s, 7
            elif p_type == 'school':
                U_sub, V_sub, W_sub, r_sub = U_sc, V_sc, W_sc, 4
            else:
                U_sub, V_sub, W_sub, r_sub = U_sb, V_sb, W_sb, 4
            
            for i in range(n):
                for js in range(m_sub):
                    j = j_start + js
                    U[i * m + j, col:col + r_sub] = U_sub[i * 2 + js, :]
            for js in range(m_sub):
                j = j_start + js
                for ks in range(p_sub):
                    k = k_start + ks
                    V[j * p + k, col:col + r_sub] = V_sub[js * p_sub + ks, :]
            for ks in range(p_sub):
                k = k_start + ks
                for i in range(n):
                    W[k * n + i, col:col + r_sub] = W_sub[ks * 2 + i, :]
            col += r_sub
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Standard block split m=2+2, p=2+2+2+1
    m_split_1 = [(0, 2), (2, 4)]
    p_split_1 = [(0, 2, 'strassen'), (2, 4, 'strassen'), (4, 6, 'strassen'), (6, 7, 'school')]
    res = _build_decomp(n, m, p, m_split_1, p_split_1, deadline)
    if res is not None:
        U, V, W, rank = res
        if _verify(U, V, W, n, m, p):
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 2: Try m=1+1+1+1 with p=2+2+2+1 (worse but check)
    m_split_2 = [(0, 1), (1, 2), (2, 3), (3, 4)]
    # For m_sub=1, we can't use 2x2 sub-tensors directly. Skip this.
    
    # Strategy 3: m=2+2, p=2+2+1+2 (reorder p blocks)
    p_split_3 = [(0, 2, 'strassen'), (2, 4, 'strassen'), (4, 5, 'school'), (5, 7, 'strassen')]
    res = _build_decomp(n, m, p, m_split_1, p_split_3, deadline)
    if res is not None:
        U, V, W, rank = res
        if _verify(U, V, W, n, m, p) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 4: m=2+2, p=1+2+2+2
    p_split_4 = [(0, 1, 'school'), (1, 3, 'strassen'), (3, 5, 'strassen'), (5, 7, 'strassen')]
    res = _build_decomp(n, m, p, m_split_1, p_split_4, deadline)
    if res is not None:
        U, V, W, rank = res
        if _verify(U, V, W, n, m, p) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 5: m=2+2, p=2+2+2+1 but with schoolbook for 2x2x2 (rank 4 each)
    p_split_5 = [(0, 2, 'school'), (2, 4, 'school'), (4, 6, 'school'), (6, 7, 'school')]
    res = _build_decomp(n, m, p, m_split_1, p_split_5, deadline)
    if res is not None:
        U, V, W, rank = res
        if _verify(U, V, W, n, m, p) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 6: Mixed - use Strassen for some, school for others
    p_split_6 = [(0, 2, 'strassen'), (2, 4, 'school'), (4, 6, 'strassen'), (6, 7, 'school')]
    res = _build_decomp(n, m, p, m_split_1, p_split_6, deadline)
    if res is not None:
        U, V, W, rank = res
        if _verify(U, V, W, n, m, p) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    if best_result is None:
        # Fallback: full schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
