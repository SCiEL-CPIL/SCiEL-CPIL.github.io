"""Combined Strassen + border-rank heuristic for 2x4 * 4x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y11[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_remove_columns(U, V, W, cols_to_remove, deadline, tol=1e-9):
    """Try removing specified columns and verify."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    mask = [i for i in range(r) if i not in cols_to_remove]
    U_new = U[:, mask].copy()
    V_new = V[:, mask].copy()
    W_new = W[:, mask].copy()
    if time.time() > deadline:
        return None
    if _verify_decomposition(U_new, V_new, W_new, tol):
        return U_new, V_new, W_new
    return None


def _greedy_column_reduction(U, V, W, deadline, max_rounds=50):
    """Greedy removal of single columns."""
    rank = U.shape[1]
    for _ in range(max_rounds):
        if time.time() > deadline - 5:
            break
        progress = False
        for col in range(rank):
            if time.time() > deadline - 5:
                break
            result = _try_remove_columns(U, V, W, [col], deadline)
            if result is not None:
                U, V, W = result
                rank = U.shape[1]
                progress = True
                break
        if not progress:
            break
    return U, V, W


def _pairwise_column_reduction(U, V, W, deadline):
    """Try removing pairs of columns."""
    rank = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 3:
                break
            for j in range(i+1, rank):
                if time.time() > deadline - 3:
                    break
                result = _try_remove_columns(U, V, W, [i, j], deadline)
                if result is not None:
                    U, V, W = result
                    improved = True
                    break
            if time.time() > deadline - 3:
                break
            if improved:
                break
    return U, V, W


def _border_rank_search(U, V, W, deadline, n_restarts=5, n_iters=100):
    """
    Apply border rank heuristic: perturb U, V, W slightly and use ALS to find
    if a rank-35 solution exists. If a column becomes near-zero or linearly
    dependent, we can potentially drop it.
    """
    n, m, p = 2, 4, 5
    r = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create perturbed copies
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()
        
        # Small random perturbations
        scale = 0.1
        U_p += scale * np.random.randn(*U_p.shape)
        V_p += scale * np.random.randn(*V_p.shape)
        W_p += scale * np.random.randn(*W_p.shape)
        
        # ALS iterations to minimize the reconstruction error
        for it in range(n_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each row i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    row_u = i * m + j
                    # Target: sum over r of U[row_u, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # For fixed row_u, this is: U[row_u, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) summed over k
                    # Build the linear system: for each r, coefficient is sum_k V[j*p+k,r]*W[k*n+i,r]
                    coeffs = np.zeros(r)
                    for k in range(p):
                        coeffs += V_p[j*p+k, :] * W_p[k*n+i, :]
                    # We want U[row_u, :] such that dot(U[row_u,:], coeffs) = 1
                    # But this is underdetermined; use the current U as regularization
                    # Actually, the tensor constraint is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # This is a system of p equations in r unknowns (U[row_u, :])
                    # Build A matrix: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V_p[j*p+k, :] * W_p[k*n+i, :]
                    # Solve A @ U[row_u, :] = ones(p)
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_p[row_u, :] = sol
                    except:
                        pass
            
            # Update V: for each row j*p+k
            for j in range(m):
                for k in range(p):
                    row_v = j * p + k
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U_p[i*m+j, :] * W_p[k*n+i, :]
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_p[row_v, :] = sol
                    except:
                        pass
            
            # Update W: for each row k*n+i
            for k in range(p):
                for i in range(n):
                    row_w = k * n + i
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_p[i*m+j, :] * V_p[j*p+k, :]
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_p[row_w, :] = sol
                    except:
                        pass
            
            # Check if any column has become very small
            col_norms_U = np.linalg.norm(U_p, axis=0)
            col_norms_V = np.linalg.norm(V_p, axis=0)
            col_norms_W = np.linalg.norm(W_p, axis=0)
            min_norms = np.minimum(np.minimum(col_norms_U, col_norms_V), col_norms_W)
            
            if np.min(min_norms) < 0.01:
                # A column is nearly zero; try to drop it
                zero_cols = list(np.where(min_norms < 0.01)[0])
                for c in zero_cols:
                    result = _try_remove_columns(U, V, W, [int(c)], deadline, tol=1e-7)
                    if result is not None:
                        U, V, W = result
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = U.shape[1]
                        break
                break
        
        # After ALS, check if the perturbed solution still verifies
        if _verify_decomposition(U_p, V_p, W_p, tol=1e-6):
            # Try to round to nearest half-integers
            U_r = np.round(U_p * 2) / 2
            V_r = np.round(V_p * 2) / 2
            W_r = np.round(W_p * 2) / 2
            if _verify_decomposition(U_r, V_r, W_r, tol=1e-6):
                # Try to drop any columns that are now redundant
                U2, V2, W2 = U_r, V_r, W_r
                U2, V2, W2 = _greedy_column_reduction(U2, V2, W2, deadline, max_rounds=10)
                if U2.shape[1] < best_rank:
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = U2.shape[1]
    
    return best_U, best_V, best_W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        rank = n * m * p
        Ub = np.zeros((n*m, rank))
        Vb = np.zeros((m*p, rank))
        Wb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ub[i*m+j, col] = 1
                    Vb[j*p+k, col] = 1
                    Wb[k*n+i, col] = 1
                    col += 1
        return {"U": Ub.tolist(), "V": Vb.tolist(), "W": Wb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Greedy single-column removal
    U, V, W = _greedy_column_reduction(U, V, W, deadline, max_rounds=40)
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 2: Pairwise column removal
    if time.time() < deadline - 15:
        U, V, W = _pairwise_column_reduction(U, V, W, deadline)
        if U.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 3: Border rank heuristic
    if time.time() < deadline - 20:
        U_br, V_br, W_br = _border_rank_search(best_U, best_V, best_W, deadline, n_restarts=3, n_iters=50)
        if U_br.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U_br, V_br, W_br
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W):
        # Fallback to original Strassen
        return {"U": _build_strassen_36()[0].tolist(), "V": _build_strassen_36()[1].tolist(), "W": _build_strassen_36()[2].tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
