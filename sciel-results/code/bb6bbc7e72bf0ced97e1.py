"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Build a valid block decomposition using Strassen for 2x2x2 blocks (rank 84).
2. Use fast ALS with pre-computed Gram matrices to converge the decomposition.
3. Attempt rank reduction by:
   a. Projecting onto lower-rank subspace and re-optimizing.
   b. Column pruning with local re-optimization.
   c. Random restarts at target rank with multiple initializations.
4. Return the best valid decomposition found within the time budget.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=tol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6 with rank 84.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _als_update_U_fast(U, V, W, T_flat, n, m, p, lam=1e-6):
    """
    Fast ALS update for U. T_flat is T reshaped to (nm, mp*pn).
    Pre-computes M = (V⊗W) reshaped to (mp*pn, r).
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # M[b, k] = V[j,k] * W[k',k] where b = j*pn + k'
    # Actually M is (mp*pn, r) where M[(j*pn+k'), k] = V[j,k] * W[k',k]
    M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
    
    # Solve (M^T M + lam*I) U^T = M^T T_flat^T
    MtM = M.T @ M + lam * np.eye(r)
    rhs = M.T @ T_flat.T
    
    try:
        U_new = np.linalg.solve(MtM, rhs).T
    except np.linalg.LinAlgError:
        U_new = np.linalg.lstsq(MtM, rhs, rcond=None)[0].T
    
    return U_new

def _als_update_V_fast(U, V, W, T_flat, n, m, p, lam=1e-6):
    """
    Fast ALS update for V. T_flat is T reshaped to (nm, mp*pn).
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    # M[a, k] = U[i,k] * W[k',k] where a = i*pn + k'
    M = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
    
    # T reshaped to (mp, nm*pn)
    T_flat_T = T_flat.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
    
    MtM = M.T @ M + lam * np.eye(r)
    rhs = M.T @ T_flat_T.T
    
    try:
        V_new = np.linalg.solve(MtM, rhs).T
    except np.linalg.LinAlgError:
        V_new = np.linalg.lstsq(MtM, rhs, rcond=None)[0].T
    
    return V_new

def _als_update_W_fast(U, V, W, T_flat, n, m, p, lam=1e-6):
    """
    Fast ALS update for W. T_flat is T reshaped to (nm, mp*pn).
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    # M[b, k] = U[i,k] * V[j,k] where b = i*mp + j
    M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
    
    # T reshaped to (pn, nm*mp)
    T_flat_T = T_flat.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
    
    MtM = M.T @ M + lam * np.eye(r)
    rhs = M.T @ T_flat_T.T
    
    try:
        W_new = np.linalg.solve(MtM, rhs).T
    except np.linalg.LinAlgError:
        W_new = np.linalg.lstsq(MtM, rhs, rcond=None)[0].T
    
    return W_new

def _als_iterate(U, V, W, T_flat, n, m, p, max_iters=100, lam=1e-6, tol=1e-10, deadline=None):
    """
    Run ALS iterations until convergence or deadline.
    """
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        U_old = U.copy()
        U = _als_update_U_fast(U, V, W, T_flat, n, m, p, lam)
        V = _als_update_V_fast(U, V, W, T_flat, n, m, p, lam)
        W = _als_update_W_fast(U, V, W, T_flat, n, m, p, lam)
        
        if np.linalg.norm(U - U_old, 'fro') < tol:
            break
    
    return U, V, W

def _try_rank_reduction(U, V, W, T_flat, n, m, p, target_rank, deadline, tol=1e-9):
    """
    Try to reduce to target_rank using ALS from current solution.
    """
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if current_rank < target_rank:
        diff = target_rank - current_rank
        U = np.hstack([U, np.zeros((nm, diff))])
        V = np.hstack([V, np.zeros((mp, diff))])
        W = np.hstack([W, np.zeros((pn, diff))])
    elif current_rank > target_rank:
        # Use SVD to project onto lower rank
        # Combine all factors into a single tensor and do truncated SVD
        # Or just take first target_rank columns
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
    
    U, V, W = _als_iterate(U, V, W, T_flat, n, m, p, max_iters=200, lam=1e-6, deadline=deadline)
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    return None

def _try_random_restart(target_rank, T_flat, n, m, p, deadline, tol=1e-9, n_restarts=10):
    """
    Try random initializations at target rank.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        U, V, W = _als_iterate(U, V, W, T_flat, n, m, p, max_iters=300, lam=1e-6, deadline=deadline)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    return None

def _try_column_pruning(U, V, W, T_flat, n, m, p, deadline, tol=1e-9):
    """
    Try to prune columns with small norms and re-optimize.
    """
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
    
    for prune_iter in range(30):
        if time.time() > deadline - 10:
            break
        if current_rank <= 60:
            break
        
        col_norms = np.sqrt(np.sum(U_curr**2, axis=0) + np.sum(V_curr**2, axis=0) + np.sum(W_curr**2, axis=0))
        
        if np.min(col_norms) < 1e-8:
            # Remove the smallest column
            min_idx = np.argmin(col_norms)
            mask = np.ones(current_rank, dtype=bool)
            mask[min_idx] = False
            
            U_test = U_curr[:, mask].copy()
            V_test = V_curr[:, mask].copy()
            W_test = W_curr[:, mask].copy()
            
            U_test, V_test, W_test = _als_iterate(U_test, V_test, W_test, T_flat, n, m, p, max_iters=100, lam=1e-6, deadline=deadline)
            
            if _verify_decomp(n, m, p, U_test, V_test, W_test, tol):
                U_curr, V_curr, W_curr = U_test, V_test, W_test
                current_rank = U_curr.shape[1]
            else:
                break
        else:
            break
    
    return U_curr, V_curr, W_curr

def _try_svd_projection(U, V, W, T_flat, n, m, p, target_rank, deadline, tol=1e-9):
    """
    Use SVD to project the current decomposition onto a lower-rank subspace.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full tensor from current decomposition
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    
    # Try different modes of unfolding
    # Mode-1: (nm, mp*pn)
    T_m1 = T_comp.reshape(nm, mp * pn)
    U_svd, s, Vt = np.linalg.svd(T_m1, full_matrices=False)
    
    # Take top target_rank components
    U_new = (U_svd[:, :target_rank] * s[:target_rank]).T  # (target_rank, nm) -> need (nm, target_rank)
    U_new = U_new.T  # (nm, target_rank)
    
    # For V and W, we need to factor the remaining part
    # This is tricky, so let's just use the SVD for U and re-optimize V, W
    V_new = np.random.randn(mp, target_rank) * 0.1
    W_new = np.random.randn(pn, target_rank) * 0.1
    
    U_new, V_new, W_new = _als_iterate(U_new, V_new, W_new, T_flat, n, m, p, max_iters=200, lam=1e-6, deadline=deadline)
    
    if _verify_decomp(n, m, p, U_new, V_new, W_new, tol):
        return U_new, V_new, W_new
    
    # Try mode-2 unfolding
    T_m2 = T_comp.transpose(1, 0, 2).reshape(mp, nm * pn)
    U_svd2, s2, Vt2 = np.linalg.svd(T_m2, full_matrices=False)
    
    V_new2 = (U_svd2[:, :target_rank] * s2[:target_rank]).T
    V_new2 = V_new2.T  # (mp, target_rank)
    
    U_new2 = np.random.randn(nm, target_rank) * 0.1
    W_new2 = np.random.randn(pn, target_rank) * 0.1
    
    U_new2, V_new2, W_new2 = _als_iterate(U_new2, V_new2, W_new2, T_flat, n, m, p, max_iters=200, lam=1e-6, deadline=deadline)
    
    if _verify_decomp(n, m, p, U_new2, V_new2, W_new2, tol):
        return U_new2, V_new2, W_new2
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Step 1: Build the block decomposition (rank 84)
    U, V, W = _block_decomposition_356()
    current_rank = U.shape[1]
    
    # Pre-compute target tensor
    T_target = _build_target_tensor(n, m, p)
    T_flat = T_target.reshape(n * m, m * p * p * n)
    
    # Verify initial decomposition
    if not _verify_decomp(n, m, p, U, V, W, tol=1e-8):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Converge the initial decomposition
    U, V, W = _als_iterate(U, V, W, T_flat, n, m, p, max_iters=50, lam=1e-6, deadline=deadline)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Step 2: Try rank reduction from 84 down
    if time.time() < deadline - 20:
        # First, try column pruning
        U_pruned, V_pruned, W_pruned = _try_column_pruning(U, V, W, T_flat, n, m, p, deadline)
        if U_pruned.shape[1] < best_rank:
            best_U, best_V, best_W = U_pruned, V_pruned, W_pruned
            best_rank = U_pruned.shape[1]
    
    # Step 3: Try direct rank reduction to various targets
    target_ranks = [75, 70, 65, 60, 55, 50, 45, 40]
    for target in target_ranks:
        if time.time() > deadline - 15:
            break
        if target >= best_rank:
            continue
        
        # Try SVD projection
        result = _try_svd_projection(best_U, best_V, best_W, T_flat, n, m, p, target, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            if U_r.shape[1] < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
                continue
        
        # Try direct ALS from current solution
        result = _try_rank_reduction(best_U, best_V, best_W, T_flat, n, m, p, target, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            if U_r.shape[1] < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Step 4: Try random restarts at best_rank and lower
    for target in [best_rank, best_rank - 5, best_rank - 10, best_rank - 15]:
        if time.time() > deadline - 10:
            break
        if target < 30:
            continue
        
        result = _try_random_restart(target, T_flat, n, m, p, deadline, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            if U_r.shape[1] < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 5:
        U_polished, V_polished, W_polished = _als_iterate(
            best_U, best_V, best_W, T_flat, n, m, p, 
            max_iters=100, lam=1e-8, deadline=deadline
        )
        if _verify_decomp(n, m, p, U_polished, V_polished, W_polished, tol=1e-10):
            best_U, best_V, best_W = U_polished, V_polished, W_polished
    
    # Final verification
    if not _verify_decomp(n, m, p, best_U, best_V, best_W, tol=1e-8):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
