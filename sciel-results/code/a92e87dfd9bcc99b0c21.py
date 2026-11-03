"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Build a valid block decomposition using Strassen for 2x2x2 blocks.
2. Use alternating least squares with rank reduction to find a lower-rank
   decomposition that still exactly decomposes the multiplication tensor.
3. Return the best valid decomposition found within the time budget.
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
        i_off += i_s * 0 + 0  # placeholder to keep structure
    
    return U_total, V_total, W_total

def _als_update_U(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update U given V, W, and target tensor T.
    For each row a of U, solve:
        sum_k U[a,k] * V[b,k] * W[c,k] = T[a,b,c] for all b,c
    This is: U[a,:] @ (V * W) = T[a,:,:].reshape(-1)
    where (V*W) is the element-wise product matrix of shape (mp*pn, rank)
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # Build the design matrix M of shape (mp*pn, r)
    # M[b*pn+c, k] = V[b,k] * W[c,k]
    M = np.zeros((mp * pn, r), dtype=np.float64)
    for b in range(mp):
        for c in range(pn):
            M[b * pn + c, :] = V[b, :] * W[c, :]
    
    U_new = np.zeros_like(U)
    for a in range(nm):
        t_a = T_target[a, :, :].reshape(-1)
        # Solve M @ x = t_a with ridge regularization
        # (M^T M + lam * I) x = M^T t_a
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_a
        try:
            U_new[a, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            U_new[a, :] = np.linalg.lstsq(M, t_a, rcond=None)[0]
    
    return U_new

def _als_update_V(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update V given U, W, and target tensor T.
    For each row b of V, solve:
        sum_k U[a,k] * V[b,k] * W[c,k] = T[a,b,c] for all a,c
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    M = np.zeros((nm * pn, r), dtype=np.float64)
    for a in range(nm):
        for c in range(pn):
            M[a * pn + c, :] = U[a, :] * W[c, :]
    
    V_new = np.zeros_like(V)
    for b in range(mp):
        t_b = T_target[:, b, :].reshape(-1)
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_b
        try:
            V_new[b, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            V_new[b, :] = np.linalg.lstsq(M, t_b, rcond=None)[0]
    
    return V_new

def _als_update_W(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update W given U, V, and target tensor T.
    For each row c of W, solve:
        sum_k U[a,k] * V[b,k] * W[c,k] = T[a,b,c] for all a,b
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    M = np.zeros((nm * mp, r), dtype=np.float64)
    for a in range(nm):
        for b in range(mp):
            M[a * mp + b, :] = U[a, :] * V[b, :]
    
    W_new = np.zeros_like(W)
    for c in range(pn):
        t_c = T_target[:, :, c].reshape(-1)
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_c
        try:
            W_new[c, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            W_new[c, :] = np.linalg.lstsq(M, t_c, rcond=None)[0]
    
    return W_new

def _als_optimize(U, V, W, T_target, n, m, p, max_iter=50, lam=0.01, tol=1e-10):
    """
    Run ALS optimization to minimize the residual.
    Returns (U, V, W, converged)
    """
    prev_err = float('inf')
    for it in range(max_iter):
        U = _als_update_U(U, V, W, T_target, n, m, p, lam)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam)
        
        T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        err = np.linalg.norm(T_comp - T_target)
        
        if abs(prev_err - err) < tol:
            return U, V, W, True
        prev_err = err
    
    return U, V, W, False

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_attempts=5, deadline=None):
    """
    Try to reduce the rank of the decomposition to target_rank.
    Uses random initialization and ALS.
    Returns (U, V, W, success)
    """
    nm, mp, pn = n * m, m * p, p * n
    T_target = _build_target_tensor(n, m, p)
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline - 5:
            return U, V, W, False
        
        # Initialize randomly
        np.random.seed(42 + attempt)
        U_init = np.random.randn(nm, target_rank) * 0.1
        V_init = np.random.randn(mp, target_rank) * 0.1
        W_init = np.random.randn(pn, target_rank) * 0.1
        
        # Run ALS
        U_opt, V_opt, W_opt, converged = _als_optimize(
            U_init, V_init, W_init, T_target, n, m, p,
            max_iter=30, lam=0.001, tol=1e-12
        )
        
        # Check if we achieved a valid decomposition
        if _verify_decomp(n, m, p, U_opt, V_opt, W_opt, tol=1e-9):
            return U_opt, V_opt, W_opt, True
    
    return U, V, W, False

def _to_half_integer(matrix):
    """Convert matrix entries to nearest half-integer."""
    return np.round(matrix * 2) / 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with the block decomposition (Strassen-based)
    U, V, W = _block_decomposition_356()
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Verify the initial decomposition
    if not _verify_decomp(n, m, p, U, V, W, tol=1e-9):
        # Fall back to schoolbook if something went wrong
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
    
    # Try to reduce the rank using ALS
    # The theoretical lower bound is unknown, but let's try progressively lower ranks
    # Start from a bit below the current rank and go down
    current_rank = best_rank
    
    for target_rank in range(max(30, current_rank - 10), 25, -1):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new, success = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p, target_rank,
            max_attempts=3, deadline=deadline
        )
        
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
    
    # Convert to half-integers if possible
    U_final = _to_half_integer(best_U)
    V_final = _to_half_integer(best_V)
    W_final = _to_half_integer(best_W)
    
    # Verify the final decomposition
    if not _verify_decomp(n, m, p, U_final, V_final, W_final, tol=1e-9):
        # If half-integer conversion broke it, keep the original
        U_final = best_U
        V_final = best_V
        W_final = best_W
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
