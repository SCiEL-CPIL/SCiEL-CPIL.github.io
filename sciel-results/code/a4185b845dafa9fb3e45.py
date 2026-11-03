"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 128
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def _max_error(U, V, W, n, m, p):
        """Compute max |sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1| over all (i,j,k)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j]
                    v = V[j * p + k]
                    w = W[k * n + i]
                    total = np.dot(u, v * w)
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def _als_step(U, V, W, n, m, p):
        """One ALS iteration: update U, then V, then W."""
        r = U.shape[1]
        
        # Update U
        U_new = U.copy()
        for idx in range(n * m):
            i = idx // m
            j = idx % m
            # Build M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            M = np.zeros((p, r))
            for k in range(p):
                M[k, :] = V[j * p + k] * W[k * n + i]
            # Solve M @ x = ones(p)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=1e-12)
                U_new[idx, :] = x
            except np.linalg.LinAlgError:
                pass
        
        # Update V
        V_new = V.copy()
        for idx in range(m * p):
            j = idx // p
            k = idx % p
            # Build M[i, c] = U[i*m+j, c] * W[k*n+i, c]
            M = np.zeros((n, r))
            for i in range(n):
                M[i, :] = U[i * m + j] * W[k * n + i]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=1e-12)
                V_new[idx, :] = x
            except np.linalg.LinAlgError:
                pass
        
        # Update W
        W_new = W.copy()
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            # Build M[j, c] = U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((m, r))
            for j in range(m):
                M[j, :] = U[i * m + j] * V[j * p + k]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=1e-12)
                W_new[idx, :] = x
            except np.linalg.LinAlgError:
                pass
        
        return U_new, V_new, W_new
    
    def _try_rank(target_rank, n, m, p, seed=42, max_iter=50, tol=1e-8):
        """Try to find a valid rank-`target_rank` decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iter):
            U, V, W = _als_step(U, V, W, n, m, p)
            if it % 10 == 0:
                err = _max_error(U, V, W, n, m, p)
                if err < tol:
                    return U, V, W, err
        
        err = _max_error(U, V, W, n, m, p)
        if err < 1e-6:
            return U, V, W, err
        return None, None, None, err
    
    deadline = time.time() + 140.0
    
    best_result = None
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For [4,4,8], try a range of ranks
    for target_rank in range(40, rank_schoolbook + 1, 5):
        if time.time() > deadline:
            break
        
        U_res, V_res, W_res, err = _try_rank(target_rank, n, m, p, seed=target_rank, max_iter=30)
        
        if U_res is not None and err < 1e-6:
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = target_rank
            break
    
    if best_rank == rank_schoolbook:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
