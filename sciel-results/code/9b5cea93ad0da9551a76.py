"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s kill

    # Kronecker-product strategy:
    # 2x4x7 = (2x2x7) ⊗ (1x2x1)
    # The 1x2x1 identity tensor has rank 2 (two rank-1 terms)
    # So if we have a rank-r decomposition of 2x2x7, we get rank 2*r for 2x4x7
    #
    # For 2x2x7, use Strassen on (0,1),(2,3),(4,5) and schoolbook on (6):
    # rank = 3*7 + 4 = 25
    # Global rank = 2 * 25 = 50
    #
    # Then aggressively search for linear dependencies to reduce rank.

    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    # Build the 2x2x7 base decomposition
    n_sub, m_sub, p_sub = 2, 2, 7
    r_base = 25
    U_base = np.zeros((n_sub * m_sub, r_base))
    V_base = np.zeros((m_sub * p_sub, r_base))
    W_base = np.zeros((p_sub * n_sub, r_base))
    
    col_offset = 0
    for k_start in [0, 2, 4]:
        for i in range(n_sub):
            for j in range(m_sub):
                U_base[i * m_sub + j, col_offset:col_offset+7] = U_s[i * m_sub + j, :]
        for j in range(m_sub):
            for k_sub in range(2):
                V_base[j * p_sub + k_start + k_sub, col_offset:col_offset+7] = V_s[j * 2 + k_sub, :]
        for k_sub in range(2):
            for i in range(n_sub):
                W_base[(k_start + k_sub) * n_sub + i, col_offset:col_offset+7] = W_s[k_sub * n_sub + i, :]
        col_offset += 7
    
    for c in range(4):
        i = c // m_sub
        j = c % m_sub
        U_base[i * m_sub + j, col_offset + c] = 1.0
        V_base[j * p_sub + 6, col_offset + c] = 1.0
        W_base[6 * n_sub + i, col_offset + c] = 1.0
    col_offset += 4
    
    if not _verify(U_base, V_base, W_base, n_sub, m_sub, p_sub):
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
    
    # Try to reduce the base 2x2x7 rank
    if time.time() < deadline - 15:
        U_base_red, V_base_red, W_base_red = _try_reduce_rank_base(U_base, V_base, W_base, n_sub, m_sub, p_sub, deadline)
        if U_base_red is not None:
            U_base, V_base, W_base = U_base_red, V_base_red, W_base_red
    
    r_base_final = U_base.shape[1]
    
    # Build the full 2x4x7 via Kronecker structure
    # The 1x2x1 identity: A is 1x1, B is 2x1, C is 1x2
    # T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k'
    # For the Kronecker: full rank = 2 * r_base
    # The two "copies" correspond to the two rows of the 2x1 middle matrix
    
    total_rank = 2 * r_base_final
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy 0: j in {0,1} (first block of middle dimension)
    # Copy 1: j in {2,3} (second block of middle dimension)
    for copy_idx in range(2):
        j_start = copy_idx * 2
        col_start = copy_idx * r_base_final
        
        for i in range(n):
            for j_sub in range(2):
                j = j_start + j_sub
                U[i * m + j, col_start:col_start + r_base_final] = U_base[i * 2 + j_sub, :]
        
        for j_sub in range(2):
            j = j_start + j_sub
            for k in range(p):
                V[j * p + k, col_start:col_start + r_base_final] = V_base[j_sub * p + k, :]
        
        for k in range(p):
            for i in range(n):
                W[k * n + i, col_start:col_start + r_base_final] = W_base[k * n + i, :]
    
    if not _verify(U, V, W, n, m, p):
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = total_rank
    
    # Try to reduce the full 2x4x7 rank
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if U_red is not None and U_red.shape[1] < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_reduce_rank_base(U, V, W, n, m, p, deadline):
    """Try to reduce the rank of a base tensor decomposition."""
    r = U.shape[1]
    
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    
    n_constraints = len(rows)
    
    # Schoolbook columns (last 4) are most likely redundant
    order = list(range(r - 1, r - 5, -1)) + list(range(r - 5, -1, -1))
    
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
        
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Try absorbing alpha into U
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
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur -= 1
            removed.add(c0)
            continue
        
        # Try absorbing alpha into V
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
    
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    
    return None, None, None

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
    order = list(range(r - 1, r - 9, -1)) + list(range(r - 9, -1, -1))
    
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
    
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    
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
    """Reserved for future use - currently unused in the Kronecker approach."""
    return None, None, None

# EVOLVE-BLOCK-END
