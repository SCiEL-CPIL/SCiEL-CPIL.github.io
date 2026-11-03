"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # Safety margin before 150s kill

    # Build the initial rank-50 decomposition
    U, V, W = _build_base_rank50(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return _fallback(n, m, p)
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Attempt rank reduction
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _reduce_rank_global(U, V, W, n, m, p, deadline - 5)
        if U_red is not None and U_red.shape[1] < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_base_rank50(n, m, p):
    # m=4, p=7. Split m into two blocks of 2: j in {0,1} and {2,3}
    # For each j-block, we have a 2x2x7 sub-problem.
    # The 2x2x7 problem is solved by:
    #   - 3 Strassen blocks for k in {0,1}, {2,3}, {4,5} (rank 7 each)
    #   - 1 Schoolbook block for k in {6} (rank 4)
    # Total rank for one j-block: 3*7 + 4 = 25
    # Total rank for both j-blocks: 2 * 25 = 50
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2, 'strassen'), (2, 4, 'strassen'), (4, 6, 'strassen'), (6, 7, 'school')]
    
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

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    # T[i, j, jp, k] = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
    # Expected: 1 if j == jp else 0
    # Note: The tensor is T[i,j,k] for the product A[i,j]*B[j,k].
    # The decomposition is sum_c U[i,j,c] V[j',k,c] W[k,i,c] = delta_{j,j'}
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _reduce_rank_global(U, V, W, n, m, p, deadline):
    r = U.shape[1]
    
    # We look for a column c0 that can be removed by modifying other columns.
    # The condition is: U_c0 V_c0 W_c0 = sum_{c != c0} alpha_c U_c V_c W_c
    # This is a linear system in alpha_c if we fix the products U_c V_c W_c.
    # Actually, the tensor equation is:
    # T = sum_c U_c V_c W_c
    # We want to see if we can write U_c0 V_c0 W_c0 as a linear combination of the others.
    # Let A_c = U_c V_c (matrix of size nm x mp). Let w_c = W_c (vector of size pn).
    # The equation is sum_{c != c0} alpha_c A_c w_c = A_c0 w_c0.
    # This is not linear in alpha_c because w_c is fixed but A_c depends on the scaling.
    # However, if we allow scaling of U_c or V_c, we can absorb alpha_c into U_c.
    # Let U'_c = alpha_c U_c. Then sum_{c != c0} U'_c V_c W_c = U_c0 V_c0 W_c.
    # This is a linear system for the elements of U'_c? No.
    # The equation is linear in the "flattened" tensors.
    # Flatten U to u (nm*r), V to v (mp*r), W to w (pn*r).
    # The tensor T is a 4-way tensor T[i, j, jp, k] = sum_c U[i,j,c] V[jp,k,c] W[k,i,c].
    # This is a multilinear form.
    # We can view it as a linear system in the columns of U, V, W jointly.
    # Let's use the approach from the previous program which seemed to work for finding dependencies.
    # It solved A alpha = b where A[i,j,jp,k, c] = U[i,j,c] V[jp,k,c] W[k,i,c].
    # This works because we are looking for alpha such that sum alpha_c T_c = T_c0.
    # Then we set U'_c = alpha_c U_c and keep V, W same.
    # This is a valid rank reduction if a solution exists.
    
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    n_constraints = len(rows)
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    r_cur = r
    removed = set()
    
    # Order columns: try removing schoolbook-like or simple columns first?
    # Or just iterate.
    # The previous code tried a specific order. Let's use a greedy approach:
    # Try each column, check if it's redundant.
    
    for _ in range(r_cur):
        if time.time() > deadline:
            break
        if r_cur <= 1:
            break
            
        # Try to find a column that can be removed
        found = False
        for c0 in range(r_cur):
            if c0 in removed:
                continue
                
            active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
            if len(active_cols) == 0:
                continue
                
            # Build system A alpha = b
            # A[idx, col_idx] = U[i,j,c] V[jp,k,c] W[k,i,c]
            # b[idx] = U[i,j,c0] V[jp,k,c0] W[k,i,c0]
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
                
            # Found a solution. Apply it.
            # Scale U for active columns by alpha.
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
                found = True
                break # Restart loop to check for new redundancies
                
        if not found:
            break
            
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    return None, None, None

def _fallback(n, m, p):
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

# EVOLVE-BLOCK-END
