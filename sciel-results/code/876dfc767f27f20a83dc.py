"""Matrix multiplication 3x4x4 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
SCHOOLBOOK_RANK = N * M * P
_SEARCH_DEADLINE = 115.0
_start_time = None


def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, target, atol=tol)


def _to_fraction_list(mat):
    """Convert a numpy array to a list of lists of Fractions for exact output."""
    result = []
    for row in mat:
        r = []
        for val in row:
            # Try to find a simple fraction
            f = round(val * 2) / 2
            if abs(val - f) < 1e-8:
                r.append(int(f) if f == int(f) else f)
            else:
                f = round(val * 4) / 4
                if abs(val - f) < 1e-8:
                    r.append(int(f) if f == int(f) else f)
                else:
                    r.append(round(val, 10))
        result.append(r)
    return result


def _alternating_minimize(target_rank, seed, max_iters=300, tol=1e-10):
    """
    Vectorized alternating minimization to find rank-target_rank decomposition.
    Returns (U, V, W) as numpy arrays or None.
    """
    global _start_time
    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n
    T = np.ones((n, m, p), dtype=np.float64)
    
    rng = np.random.default_rng(seed)
    
    U = rng.normal(size=(nm, target_rank)) * 0.5
    V = rng.normal(size=(mp, target_rank)) * 0.5
    W = rng.normal(size=(pn, target_rank)) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - _start_time > _SEARCH_DEADLINE - 5:
            break
        
        # Update U: for each row u, solve (V*W elementwise) @ U[u,:] = T[u,:,:].flatten()
        # A_u[i, r] = V[v_i, r] * W[w_i, r] where i = v*pn + w
        # b_u = T[u, :, :].flatten()
        for u in range(nm):
            # Extract i, j from u
            i = u // m
            j = u % m
            # Build A: (mp*pn, R)
            # A[v*pn + w, r] = V[v, r] * W[w, r]
            # But we only need entries where the corresponding T[i,j,k] = 1
            # Actually, for the matmul tensor, T[i,j,k]=1 for all i,j,k
            # So b_u = ones(mp*pn)
            # A[v*pn+w, r] = V[v,r]*W[w,r]
            P = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, R)
            A = P.reshape(mp * pn, target_rank)
            b = np.ones(mp * pn)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol
            except:
                pass
        
        # Update V: for each row v, solve (U*W elementwise) @ V[v,:] = T[:,v,:].flatten()
        for v in range(mp):
            j = v // p
            k = v % p
            P = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, R)
            A = P.reshape(nm * pn, target_rank)
            b = np.ones(nm * pn)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol
            except:
                pass
        
        # Update W: for each row w, solve (U*V elementwise) @ W[w,:] = T[:,:,w].flatten()
        for w in range(pn):
            k = w // n
            i = w % n
            P = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, R)
            A = P.reshape(nm * mp, target_rank)
            b = np.ones(nm * mp)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol
            except:
                pass
        
        # Check convergence
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None


def _polish(U, V, W, max_iters=100, tol=1e-12):
    """High-precision polish pass to tighten the solution."""
    global _start_time
    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n
    R = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - _start_time > _SEARCH_DEADLINE - 3:
            break
        
        # Update U
        for u in range(nm):
            P = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            A = P.reshape(mp * pn, R)
            b = np.ones(mp * pn)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol
            except:
                pass
        
        # Update V
        for v in range(mp):
            P = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            A = P.reshape(nm * pn, R)
            b = np.ones(nm * pn)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol
            except:
                pass
        
        # Update W
        for w in range(pn):
            P = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            A = P.reshape(nm * mp, R)
            b = np.ones(nm * mp)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol
            except:
                pass
        
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err


def _try_rank_with_restarts(target_rank, num_restarts=30, max_iters=200):
    """Try multiple random restarts for a given target rank."""
    global _start_time
    for seed in range(num_restarts):
        if time.time() - _start_time > _SEARCH_DEADLINE - 10:
            break
        result = _alternating_minimize(target_rank, seed=seed, max_iters=max_iters)
        if result is not None:
            U, V, W = result
            # Polish
            U, V, W, err = _polish(U, V, W, max_iters=50, tol=1e-10)
            if _verify(U, V, W, tol=1e-6):
                return U, V, W
    return None


def _greedy_rank_reduce():
    """
    Try to find a reduced-rank decomposition using alternating minimization with multiple restarts.
    """
    global _start_time
    n, m, p = N, M, P
    rank0 = n * m * p
    
    best_rank = rank0
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from rank0-1 down to 40
    for target_rank in range(rank0 - 1, 39, -1):
        if time.time() - _start_time > _SEARCH_DEADLINE - 15:
            break
        
        # Use more restarts for lower ranks (harder problems)
        num_restarts = 20 if target_rank > 50 else 30
        max_iters = 150 if target_rank > 50 else 200
        result = _try_rank_with_restarts(target_rank, num_restarts=num_restarts, max_iters=max_iters)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    if best_U is not None:
        # Round to nearest integer/half-integer
        U_mat = np.round(best_U * 2) / 2
        V_mat = np.round(best_V * 2) / 2
        W_mat = np.round(best_W * 2) / 2
        if _verify(U_mat, V_mat, W_mat, tol=1e-8):
            return U_mat, V_mat, W_mat
        # Try without rounding
        if _verify(best_U, best_V, best_W, tol=1e-8):
            return best_U, best_V, best_W
        # Try finer rounding
        U_mat2 = np.round(best_U * 10) / 10
        V_mat2 = np.round(best_V * 10) / 10
        W_mat2 = np.round(best_W * 10) / 10
        if _verify(U_mat2, V_mat2, W_mat2, tol=1e-8):
            return U_mat2, V_mat2, W_mat2
    
    return None


def _schoolbook():
    """Return the schoolbook decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W


def construct():
    global _start_time
    _start_time = time.time()
    
    result = None
    try:
        result = _greedy_rank_reduce()
    except Exception:
        result = None
    
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        U_list = _to_fraction_list(U)
        V_list = _to_fraction_list(V)
        W_list = _to_fraction_list(W)
        if _verify(U_list, V_list, W_list, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook()
    U_list = _to_fraction_list(U)
    V_list = _to_fraction_list(V)
    W_list = _to_fraction_list(W)
    return {"U": U_list, "V": V_list, "W": W_list}
