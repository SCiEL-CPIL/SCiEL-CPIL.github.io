"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Apply Strassen's rank-7 algorithm for 2x2 matrix multiplication
independently to each of the 5 column-pairs (blocks) of the 2x10 matrix B.
This gives a total rank of 5 * 7 = 35.

The decomposition is exact and uses only integer entries.

Refinement: Attempt to reduce rank by identifying linear dependencies
across the 5 independent column blocks using Gaussian elimination on the
bilinear coefficient structure.
"""
import time
import numpy as np


def _build_strassen_rank35():
    """
    Build the rank-35 Strassen decomposition for 2x2 * 2x10 multiplication.
    
    U: shape (4, 35) - factors for A (2x2, flattened row-major)
    V: shape (20, 35) - factors for B (2x10, flattened row-major)
    W: shape (20, 35) - factors for C (2x10, flattened with W[k*n+i, r] convention)
    """
    n, m, p = 2, 2, 10
    rank = 35
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    for t in range(5):
        base = t * 7
        e_idx = 0 * p + 2 * t
        f_idx = 0 * p + 2 * t + 1
        g_idx = 1 * p + 2 * t
        h_idx = 1 * p + 2 * t + 1
        
        c00 = 4 * t
        c01 = 4 * t + 2
        c10 = 4 * t + 1
        c11 = 4 * t + 3
        
        # M1 = (a+d)(e+h)
        col = base
        U[0, col] = 1
        U[3, col] = 1
        V[e_idx, col] = 1
        V[h_idx, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        
        # M2 = (c+d)e
        col = base + 1
        U[2, col] = 1
        U[3, col] = 1
        V[e_idx, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1
        
        # M3 = a(f-h)
        col = base + 2
        U[0, col] = 1
        V[f_idx, col] = 1
        V[h_idx, col] = -1
        W[c01, col] = 1
        W[c11, col] = 1
        
        # M4 = d(g-e)
        col = base + 3
        U[3, col] = 1
        V[g_idx, col] = 1
        V[e_idx, col] = -1
        W[c00, col] = 1
        W[c10, col] = 1
        
        # M5 = (a+b)h
        col = base + 4
        U[0, col] = 1
        U[1, col] = 1
        V[h_idx, col] = 1
        W[c00, col] = -1
        W[c01, col] = 1
        
        # M6 = (c-a)(e+f)
        col = base + 5
        U[2, col] = 1
        U[0, col] = -1
        V[e_idx, col] = 1
        V[f_idx, col] = 1
        W[c11, col] = 1
        
        # M7 = (b-d)(g+h)
        col = base + 6
        U[1, col] = 1
        U[3, col] = -1
        V[g_idx, col] = 1
        V[h_idx, col] = 1
        W[c00, col] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W, n=2, m=2, p=10, seed=42):
    """Fast vectorized verification of the tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        
        alpha = np.sum(A_test * u_mat, axis=1)
        beta = np.sum(B_test * v_mat, axis=0)
        
        C_computed += np.outer(alpha, beta) * w_mat.T
    
    return np.allclose(C_computed, C_expected, atol=1e-9)


def _verify_tensor_exact(U, V, W, n=2, m=2, p=10, tol=1e-9):
    """Verify by checking the full tensor reconstruction."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    nm, mp, pn = n * m, m * p, p * n
    T_recon = np.zeros((nm, mp, pn))
    for k in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, k], V_arr[:, k], W_arr[:, k])
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                T_target[a, b, c] = 1.0
    
    return np.max(np.abs(T_recon - T_target)) < tol


def _try_reduce_rank(U, V, W, n=2, m=2, p=10, max_attempts=200, deadline=None):
    """
    Try to reduce rank by finding linear dependencies among columns.
    Returns (U, V, W, new_rank) or original if no reduction found.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    cur_rank = U_arr.shape[1]
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_r = cur_rank
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        r = best_r
        
        # Try to eliminate one column at a time
        if r <= 1:
            break
        
        # Pick a random column to try to eliminate
        col_to_elim = np.random.randint(0, r)
        
        # Try to express column col_to_elim as linear combo of others
        mask = [i for i in range(r) if i != col_to_elim]
        
        # Use U to find the combination
        A_mat = U_try[:, mask]
        b_vec = U_try[:, col_to_elim]
        
        try:
            x, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        # Check if U is well-fit
        fit_err_U = np.linalg.norm(A_mat @ x - b_vec)
        if fit_err_U > 1e-6:
            continue
        
        # Verify with V
        V_fit_err = np.linalg.norm(V_try[:, mask] @ x - V_try[:, col_to_elim])
        if V_fit_err > 1e-6:
            continue
        
        # Verify with W
        W_fit_err = np.linalg.norm(W_try[:, mask] @ x - W_try[:, col_to_elim])
        if W_fit_err > 1e-6:
            continue
        
        # All three fit - eliminate this column
        new_U = np.delete(U_try, col_to_elim, axis=1)
        new_V = np.delete(V_try, col_to_elim, axis=1)
        new_W = np.delete(W_try, col_to_elim, axis=1)
        
        r -= 1
        if r < best_r:
            # Verify the decomposition
            if _verify_tensor_exact(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_r = r
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_r


def construct():
    """
    Construct the tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
        U: (4, rank), V: (20, rank), W: (20, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Build the known rank-35 Strassen solution
    U, V, W = _build_strassen_rank35()
    
    # Fast verification of base solution
    try:
        is_valid = _verify_decomposition(U, V, W)
    except Exception:
        is_valid = True
    
    if not is_valid:
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank using linear dependency search
    try:
        U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(
            U, V, W, n=2, m=2, p=10, max_attempts=300, deadline=deadline
        )
        if new_rank < 35:
            U, V, W = U_reduced, V_reduced, W_reduced
    except Exception:
        pass  # Keep the rank-35 solution
    
    # Final verification
    try:
        if not _verify_decomposition(U, V, W):
            # Fall back to rank-35 if reduced version is invalid
            U, V, W = _build_strassen_rank35()
    except Exception:
        U, V, W = _build_strassen_rank35()
    
    elapsed = time.time() - start_time
    return {"U": U, "V": V, "W": W}
