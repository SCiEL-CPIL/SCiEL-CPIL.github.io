"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s kill

    # Block decomposition: m=4 split into two 2-blocks, p=7 split into 2+2+2+1
    # Each 2x2x2 uses Strassen (rank 7), each 2x2x1 uses schoolbook (rank 4)
    # Total: 2 * (3*7 + 4) = 50

    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
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
    
    # Verify the rank-50 solution
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank by finding linear dependencies
        best_U, best_V, best_W = U, V, W
        if time.time() < deadline - 10:
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, deadline)
            if U_red is not None:
                best_U, best_V, best_W = U_red, V_red, W_red
        
        # Also try a direct shifted-Strassen construction
        if time.time() < deadline - 20:
            U_shift, V_shift, W_shift = _try_shifted_strassen(n, m, p, deadline)
            if U_shift is not None and U_shift.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = U_shift, V_shift, W_shift
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    r = U.shape[1]
    
    # Build the constraint rows
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    
    n_constraints = len(rows)
    
    # Try eliminating columns iteratively, starting from most likely redundant
    # Schoolbook columns are most likely redundant
    # We will try a few different orders to increase the chance of finding a dependency
    orders = [
        list(range(r - 1, r - 9, -1)) + list(range(r - 9, -1, -1)),
        list(range(r - 1, -1, -1)),
        list(range(r))
    ]
    
    best_U, best_V, best_W = U, V, W
    best_rank = r
    
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        r_cur = r
        removed = set()
        
        for c0 in order:
            if time.time() > deadline - 5:
                break
            if c0 in removed:
                continue
            
            # Build the system: find alpha such that
            # sum_{c not in removed, c != c0} alpha_c * U_c * V_c * W_c = U_c0 * V_c0 * W_c0
            active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
            if len(active_cols) == 0:
                continue
            
            A = np.zeros((n_constraints, len(active_cols)))
            b = np.zeros(n_constraints)
            
            for col_idx, c in enumerate(active_cols):
                for idx2, (i, j, jp, k) in enumerate(rows):
                    A[idx2, col_idx] = U_cur[i * m + j, c] * V_cur[jp * p + k, c] * W_cur[k * n + i, c]
            
            for idx2, (i, j, jp, k) in enumerate(rows):
                b[idx2] = U_cur[i * m + j, c0] * V_cur[jp * p + k, c0] * W_cur[k * n + i, c0]
            
            # Solve least squares
            try:
                alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is accurate
            if len(residuals) > 0:
                if residuals[0] > 1e-8:
                    continue
            
            # Verify the solution by constructing new matrices
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
            
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p):
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                r_cur -= 1
                removed.add(c0)
                continue
            
            # Try with different scaling: put alpha in V instead
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
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                r_cur -= 1
                removed.add(c0)
                continue
        
        if r_cur < best_rank:
            best_U = U_cur[:, :r_cur]
            best_V = V_cur[:, :r_cur]
            best_W = W_cur[:, :r_cur]
            best_rank = r_cur
            
    if best_rank < r:
        return best_U, best_V, best_W
    
    return None, None, None

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

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
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

def _try_shifted_strassen(n, m, p, deadline):
    """Try a shifted-Strassen decomposition exploiting overlapping 2x2 blocks in B."""
    # This function is kept from the original but is less likely to improve on the main reduction.
    # We return None to save time, as the main reduction is more promising.
    return None, None, None

# EVOLVE-BLOCK-END
