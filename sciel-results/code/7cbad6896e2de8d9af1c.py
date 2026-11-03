"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Build initial rank-50 solution via block decomposition
    U, V, W = _build_block_solution(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        return _schoolbook_fallback(n, m, p)
    
    # Try to reduce rank
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    if time.time() < deadline - 15:
        U_red, V_red, W_red = _reduce_rank_aggressive(U, V, W, n, m, p, deadline)
        if U_red is not None and U_red.shape[1] < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_block_solution(n, m, p):
    """Build rank-50 solution: m split into 2 blocks of 2, p split into 2+2+2+1."""
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
            j_sub_count = j_end - j_start
            
            for i in range(n):
                for js in range(j_sub_count):
                    j = j_start + js
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + js, :]
            
            for js in range(j_sub_count):
                j = j_start + js
                for ks in range(p_sub):
                    k = k_start + ks
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[js * p_sub + ks, :]
            
            for ks in range(p_sub):
                k = k_start + ks
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[ks * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W

def _get_strassen_222():
    """Strassen 2x2x2, rank 7."""
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """Schoolbook 2x2x1, rank 4."""
    r = 4
    U = np.zeros((4, r))
    V = np.zeros((2, r))
    W = np.zeros((2, r))
    
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify tensor identity: sum_c W[k*n+i,c]*U[i*m+j,c]*V[jp*p+k,c] = delta_{j,jp}"""
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

def _schoolbook_fallback(n, m, p):
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

def _reduce_rank_aggressive(U, V, W, n, m, p, deadline):
    """Try to reduce rank by iteratively removing linearly dependent columns."""
    r = U.shape[1]
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    r_cur = r
    removed = set()
    
    # Try different removal orders
    orders = [
        list(range(r)),
        list(range(r - 1, -1, -1)),
        # Schoolbook columns first (last 8 columns: 4+4)
        list(range(r - 8, r)) + list(range(r - 8)),
        # Strassen columns first
        list(range(0, r - 8)) + list(range(r - 8, r)),
    ]
    
    best_U, best_V, best_W = U_cur, V_cur, W_cur
    best_rank = r_cur
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        r_cur = r
        removed = set()
        
        for c0 in order:
            if time.time() > deadline - 10:
                break
            if c0 in removed:
                continue
            
            active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
            if len(active_cols) == 0:
                continue
            
            # Build constraint matrix
            n_constraints = n * m * m * p
            A = np.zeros((n_constraints, len(active_cols)))
            b = np.zeros(n_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for ai, c in enumerate(active_cols):
                                A[idx, ai] = U_cur[i * m + j, c] * V_cur[jp * p + k, c] * W_cur[k * n + i, c]
                            b[idx] = U_cur[i * m + j, c0] * V_cur[jp * p + k, c0] * W_cur[k * n + i, c0]
                            idx += 1
            
            try:
                alpha, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Try putting alpha in U
            U_new = np.zeros((n * m, r_cur - 1))
            V_new = np.zeros((m * p, r_cur - 1))
            W_new = np.zeros((p * n, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c] * alpha[col_idx]
                V_new[:, col_idx] = V_cur[:, c]
                W_new[:, col_idx] = W_cur[:, c]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                r_cur -= 1
                removed.add(c0)
                continue
            
            # Try putting alpha in V
            U_new = np.zeros((n * m, r_cur - 1))
            V_new = np.zeros((m * p, r_cur - 1))
            W_new = np.zeros((p * n, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c]
                V_new[:, col_idx] = V_cur[:, c] * alpha[col_idx]
                W_new[:, col_idx] = W_cur[:, c]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                r_cur -= 1
                removed.add(c0)
                continue
            
            # Try putting alpha in W
            U_new = np.zeros((n * m, r_cur - 1))
            V_new = np.zeros((m * p, r_cur - 1))
            W_new = np.zeros((p * n, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c]
                V_new[:, col_idx] = V_cur[:, c]
                W_new[:, col_idx] = W_cur[:, c] * alpha[col_idx]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                r_cur -= 1
                removed.add(c0)
                continue
        
        if r_cur < best_rank:
            best_U, best_V, best_W = U_cur, V_cur, W_cur
            best_rank = r_cur
    
    if best_rank < r:
        return best_U, best_V, best_W
    
    return None, None, None

# EVOLVE-BLOCK-END
