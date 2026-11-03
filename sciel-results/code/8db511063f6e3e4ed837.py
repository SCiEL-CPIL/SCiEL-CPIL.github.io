"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

_DECOMP_CACHE = None

def construct():
    global _DECOMP_CACHE
    if _DECOMP_CACHE is not None:
        return _DECOMP_CACHE

    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120

    # Get the rank-50 baseline
    U50, V50, W50 = _get_block_decomp(n, m, p)
    
    if not _verify(U50, V50, W50, n, m, p):
        # Fallback schoolbook
        return _schoolbook(n, m, p)
    
    # Try to reduce to rank 49
    if time.time() < deadline - 30:
        result = _try_reduce_to_rank_49(U50, V50, W50, n, m, p, deadline)
        if result is not None:
            U49, V49, W49 = result
            if _verify(U49, V49, W49, n, m, p):
                _DECOMP_CACHE = {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
                return _DECOMP_CACHE
    
    _DECOMP_CACHE = {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    return _DECOMP_CACHE

def _verify(U, V, W, n, m, p):
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

def _schoolbook(n, m, p):
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

def _get_block_decomp(n, m, p):
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2, 'strassen'), (2, 4, 'strassen'), (4, 6, 'strassen'), (6, 7, 'school')]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            
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
    
    return U, V, W

def _try_reduce_to_rank_49(U50, V50, W50, n, m, p, deadline):
    r50 = U50.shape[1]
    r49 = r50 - 1
    
    for c_remove in range(r50):
        if time.time() > deadline - 10:
            break
        
        # Zero out column c_remove in U and V
        U49 = U50.copy()
        V49 = V50.copy()
        U49[:, c_remove] = 0
        V49[:, c_remove] = 0
        
        # Solve for W49 linearly
        cols49 = [c for c in range(r50) if c != c_remove]
        
        num_unknowns = p * n * r49
        num_eqs = n * m * m * p
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for c_idx, c in enumerate(cols49):
                            unknown_idx = (k * n + i) * r49 + c_idx
                            A[eq_idx, unknown_idx] = U49[i*m+j, c] * V49[jp*p+k, c]
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        eq_idx += 1
        
        # Solve using least squares
        W49_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Construct W49
        W49 = np.zeros((p * n, r50))
        for k in range(p):
            for i in range(n):
                for c_idx, c in enumerate(cols49):
                    W49[k*n+i, c] = W49_flat[(k*n+i)*r49 + c_idx]
        
        # Verify
        if _verify(U49, V49, W49, n, m, p):
            return U49, V49, W49
    
    return None

def _get_strassen_222():
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
