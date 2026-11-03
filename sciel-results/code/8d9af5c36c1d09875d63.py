"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Build a valid block decomposition using Strassen for 2x2x2 blocks.
2. Use alternating least squares with rank reduction to find a lower-rank
   decomposition that still exactly decomposes the multiplication tensor.
3. Add sparse linear algebra refinement: use L1-regularized ALS to sparsify
   the factor matrices, then attempt to round to half-integers and verify.
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

def _als_update_U(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update U given V, W, and target tensor T.
    Optimized using pre-computed outer products.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # M[b*pn+c, k] = V[b, k] * W[c, k]
    # We can compute M.T @ M more efficiently or just build M.
    # M shape: (mp*pn, r)
    # T_target shape: (nm, mp, pn)
    
    # Construct M efficiently
    # V is (mp, r), W is (pn, r)
    # Outer product of V and W along columns?
    # M_flat = (V[:, None, :] * W[None, :, :]).reshape(mp*pn, r)
    M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
    
    # Precompute M.T @ M and M.T @ T_target slices
    # T_target[a] is (mp, pn) -> reshape to (mp*pn)
    # We want to solve for each a: M.T @ M @ U_new[a] = M.T @ T_target[a].flat
    
    MtM = M.T @ M + lam * np.eye(r)
    MtM_inv = np.linalg.inv(MtM)
    
    # T_target_flat: (nm, mp*pn)
    T_flat = T_target.reshape(nm, mp * pn)
    
    # M.T @ T_flat.T -> (r, nm)
    # Then U_new = (M.T @ T_flat.T).T @ MtM_inv ? No.
    # U_new[a] = MtM_inv @ (M.T @ T_flat[a])
    # Let Y = M.T @ T_flat.T -> (r, nm)
    # U_new = (MtM_inv @ Y).T -> (nm, r)
    
    Y = M.T @ T_flat.T
    U_new = (MtM_inv @ Y).T
    
    return U_new

def _als_update_V(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update V given U, W, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    # M[a*pn+c, k] = U[a, k] * W[c, k]
    M = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    MtM_inv = np.linalg.inv(MtM)
    
    # T_target[:, b, :] is (nm, pn) -> reshape to (nm*pn)
    # T_flat_T: (mp, nm*pn) ?
    # T_target shape (nm, mp, pn). We want for each b: M.T @ T_target[:, b, :].flat
    # T_target.transpose(1, 0, 2) -> (mp, nm, pn)
    T_flat_T = T_target.transpose(1, 0, 2).reshape(mp, nm * pn)
    
    Y = M.T @ T_flat_T.T # (r, mp)
    V_new = (MtM_inv @ Y).T # (mp, r)
    
    return V_new

def _als_update_W(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update W given U, V, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    # M[a*mp+b, k] = U[a, k] * V[b, k]
    M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    MtM_inv = np.linalg.inv(MtM)
    
    # T_target[:, :, c] is (nm, mp) -> reshape to (nm*mp)
    # T_target.transpose(2, 0, 1) -> (pn, nm, mp)
    T_flat_T = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    Y = M.T @ T_flat_T.T # (r, pn)
    W_new = (MtM_inv @ Y).T # (pn, r)
    
    return W_new

def _sparse_als_update_U(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for U with L1 regularization to promote sparsity.
    """
    U_dens = _als_update_U(U, V, W, T_target, n, m, p, lam)
    return np.sign(U_dens) * np.maximum(np.abs(U_dens) - l1_weight, 0.0)

def _sparse_als_update_V(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for V with L1 regularization.
    """
    V_dens = _als_update_V(U, V, W, T_target, n, m, p, lam)
    return np.sign(V_dens) * np.maximum(np.abs(V_dens) - l1_weight, 0.0)

def _sparse_als_update_W(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for W with L1 regularization.
    """
    W_dens = _als_update_W(U, V, W, T_target, n, m, p, lam)
    return np.sign(W_dens) * np.maximum(np.abs(W_dens) - l1_weight, 0.0)

def _round_to_half_int(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_sparse_refinement(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Try sparse ALS refinement to find a sparser (and potentially lower-rank)
    decomposition. Returns (U, V, W) if successful, else None.
    Uses an aggressive pruning strategy: zero out small columns and re-optimize.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    # Strategy:
    # 1. Run a few iterations of dense ALS to converge to a local minimum.
    # 2. Identify columns with small norm (in the combined factor space) and zero them out.
    # 3. Re-run ALS on the reduced rank.
    # 4. Repeat.
    
    U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
    current_rank = r
    
    # Initial convergence
    for _ in range(5):
        if time.time() > deadline - 30: break
        U_curr = _als_update_U(U_curr, V_curr, W_curr, T_target, n, m, p, lam=1e-6)
        V_curr = _als_update_V(U_curr, V_curr, W_curr, T_target, n, m, p, lam=1e-6)
        W_curr = _als_update_W(U_curr, V_curr, W_curr, T_target, n, m, p, lam=1e-6)
        
    if _verify_decomp(n, m, p, U_curr, V_curr, W_curr, tol):
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        best_rank = current_rank

    # Iterative pruning
    for prune_iter in range(10):
        if time.time() > deadline - 20: break
        if current_rank <= 60: break # Don't go too low blindly
        
        # Compute column norms
        # We want to remove columns that contribute little.
        # A proxy for importance is the norm of the column in U, V, W combined.
        col_norms = np.sqrt(np.sum(U_curr**2, axis=0) + np.sum(V_curr**2, axis=0) + np.sum(W_curr**2, axis=0))
        
        # Find the smallest non-zero norm
        # We only consider columns that are currently "active" (non-zero)
        active_mask = col_norms > 1e-8
        if not np.any(active_mask): break
        
        active_norms = col_norms[active_mask]
        if len(active_norms) == 0: break
        
        # Try to remove the column with the smallest norm
        # But be careful: removing a column might break the decomposition if it's essential.
        # We will test the removal.
        
        # Sort active columns by norm
        active_indices = np.where(active_mask)[0]
        sorted_indices = active_indices[np.argsort(active_norms)]
        
        removed_any = False
        for idx in sorted_indices[:5]: # Try top 5 smallest
            if time.time() > deadline - 15: break
            
            # Create a mask that excludes this index
            new_mask = np.ones(current_rank, dtype=bool)
            new_mask[idx] = False
            
            U_test = U_curr[:, new_mask]
            V_test = V_curr[:, new_mask]
            W_test = W_curr[:, new_mask]
            
            # Re-optimize the reduced rank decomposition
            # Run a few ALS iterations
            for _ in range(10):
                if time.time() > deadline - 15: break
                U_test = _als_update_U(U_test, V_test, W_test, T_target, n, m, p, lam=1e-6)
                V_test = _als_update_V(U_test, V_test, W_test, T_target, n, m, p, lam=1e-6)
                W_test = _als_update_W(U_test, V_test, W_test, T_target, n, m, p, lam=1e-6)
                
            if _verify_decomp(n, m, p, U_test, V_test, W_test, tol):
                # Success! We reduced the rank.
                U_curr, V_curr, W_curr = U_test, V_test, W_test
                current_rank = U_curr.shape[1]
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                best_rank = current_rank
                removed_any = True
                break # Break out of trying other indices, restart pruning loop
        
        if not removed_any:
            break # Couldn't remove any more columns
    
    return best_U, best_V, best_W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None, tol=1e-9):
    """
    Try to reduce the rank to target_rank using ALS.
    Returns (U, V, W) if successful, else None.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    if U.shape[1] < target_rank:
        diff = target_rank - U.shape[1]
        U = np.hstack([U, np.zeros((nm, diff))])
        V = np.hstack([V, np.zeros((mp, diff))])
        W = np.hstack([W, np.zeros((pn, diff))])
    else:
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.001)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.001)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.001)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    
    return None

def _high_precision_polish(U, V, W, n, m, p, deadline, tol=1e-12):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    """
    T_target = _build_target_tensor(n, m, p)
    r = U.shape[1]
    
    for it in range(50):
        if time.time() > deadline - 5:
            break
        
        U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.0001)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.0001)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.0001)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Tighter safety margin for faster ALS
    
    # Step 1: Build the block decomposition (rank 84)
    U, V, W = _block_decomposition_356()
    current_rank = U.shape[1]
    
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Step 2: Sparse linear algebra refinement (Primary Strategy)
    # This is now the main workhorse due to optimized ALS
    if time.time() < deadline - 10:
        sparse_result = _try_sparse_refinement(best_U, best_V, best_W, n, m, p, deadline)
        if sparse_result is not None:
            U_s, V_s, W_s = sparse_result
            if U_s.shape[1] < best_rank:
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = U_s.shape[1]
    
    # Step 3: High-precision polish
    if time.time() < deadline - 5:
        polished = _high_precision_polish(best_U, best_V, best_W, n, m, p, deadline)
        if polished is not None:
            best_U, best_V, best_W = polished
    
    # Final verification
    if not _verify_decomp(n, m, p, best_U, best_V, best_W, tol=1e-8):
        # Fallback to schoolbook
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
