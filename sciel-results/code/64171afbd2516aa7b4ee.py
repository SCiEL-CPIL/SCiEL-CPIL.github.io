"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Precompute index mappings for vectorized operations
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # For each r: U[:,r] * V[:,r] * W[:,r] should reconstruct the tensor
    
    def verify(U, V, W, r):
        """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_ld = np.array(U, dtype=np.longdouble)
        V_ld = np.array(V, dtype=np.longdouble)
        W_ld = np.array(W, dtype=np.longdouble)
        # Reshape: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # W3 is indexed [k, i, r] so subscript is kir
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.all(np.abs(result - 1.0) < 1e-8)

    def snap_half(M):
        return np.round(np.array(M) * 2) / 2

    def compute_error(U, V, W, r):
        """Compute sum of squared errors."""
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return float(np.sum((result - 1.0)**2))

    def als_decompose(r, max_iter=200, tol=1e-14, seed=42):
        """ALS with vectorized updates."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r).astype(np.float64)
        V = rng.randn(m * p, r).astype(np.float64)
        W = rng.randn(p * n, r).astype(np.float64)
        # Normalize
        U /= np.sqrt(np.sum(U**2))
        V /= np.sqrt(np.sum(V**2))
        W /= np.sqrt(np.sum(W**2))
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), solve M^T u = 1 where M[k,:] = V3[j,k,:] * W3[k,i,:]
            # M is (p, r), target is ones(p)
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # M[k, r_idx] = V3[j, k, r_idx] * W3[k, i, r_idx]
                    M = V3[j, :, :] * W3[:, i, :]  # (p, r)
                    # Solve M^T x = 1
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = sol
                    except:
                        pass
            U = U_new
            U3 = U.reshape(n, m, r)
            
            # Update V: for each (j,k), solve M^T v = 1 where M[i,:] = U3[i,j,:] * W3[k,i,:]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]  # (n, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = sol
                    except:
                        pass
            V = V_new
            V3 = V.reshape(m, p, r)
            
            # Update W: for each (k,i), solve M^T w = 1 where M[j,:] = U3[i,j,:] * V3[j,k,:]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]  # (m, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = sol
                    except:
                        pass
            W = W_new
            W3 = W.reshape(p, n, r)
            
            err = compute_error(U, V, W, r)
            if err < prev_err:
                prev_err = err
            if err < tol:
                break
            if err < 1e-15:
                break
            # Early stopping if no improvement for 10 iterations
            if it > 0 and err >= prev_err * (1 - 1e-10) and it > 15:
                break
        
        return U, V, W, prev_err

    def polish(U, V, W, r, max_iter=100):
        """High-precision polish using longdouble."""
        U_ld = U.astype(np.longdouble)
        V_ld = V.astype(np.longdouble)
        W_ld = W.astype(np.longdouble)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p, dtype=np.longdouble), rcond=None)
                        U3[i, j, :] = sol
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n, dtype=np.longdouble), rcond=None)
                        V3[j, k, :] = sol
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m, dtype=np.longdouble), rcond=None)
                        W3[k, i, :] = sol
                    except:
                        pass
            
            # Check convergence
            result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = float(np.sum((result - 1.0)**2))
            if err < 1e-20:
                break
            if err < prev_err * (1 - 1e-12):
                prev_err = err
            else:
                # No significant improvement, try one more round then stop
                if it > 10:
                    break
        
        U_ld = U3.reshape(n * m, r)
        V_ld = V3.reshape(m * p, r)
        W_ld = W3.reshape(p * n, r)
        return U_ld, V_ld, W_ld

    # Search strategy: try ranks from low to high with time budgeting
    # For 3x3*3x7, schoolbook is 63.
    # Hint suggests rank 28 is possible (24 for 3x3 + 4 for extension).
    # We will try ranks from 28 to 63.
    best_result = None
    best_rank = 63
    
    # Rank candidates: try from low to high
    ranks_to_try = list(range(28, 64))
    
    # Allocate more time to lower ranks initially
    time_per_rank = (deadline - start_time - 15) / len(ranks_to_try)
    
    # Give more time to lower ranks
    time_weights = [1.0 / (i + 1) for i in range(len(ranks_to_try))]
    total_weight = sum(time_weights)
    available_time = deadline - start_time - 15
    time_per_rank = [available_time * w / total_weight for w in time_weights]
    
    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline - 15:
            break
        
        rank_deadline = time.time() + time_per_rank[idx]
        found = False
        
        # Try multiple random restarts
        # Estimate time per restart roughly
        n_restarts = max(1, int((rank_deadline - time.time()) / 3))
        n_restarts = min(n_restarts, 3)
        
        for restart in range(n_restarts):
            if time.time() > rank_deadline - 2:
                break
            
            # Reduce max_iter for speed, rely on multiple restarts
            U, V, W, err = als_decompose(r, max_iter=50, seed=42 + r * 100 + restart)
            
            if err < 1e-10:
                # Polish
                U_ld, V_ld, W_ld = polish(U, V, W, r)
                
                # Try half-integer snap
                U_snap = snap_half(U_ld)
                V_snap = snap_half(V_ld)
                W_snap = snap_half(W_ld)
                
                if verify(U_snap, V_snap, W_snap, r):
                    best_rank = r
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    found = True
                    break
                
                # Try without snap
                if verify(U_ld, V_ld, W_ld, r):
                    best_rank = r
                    best_result = {
                        "U": U_ld.tolist(),
                        "V": V_ld.tolist(),
                        "W": W_ld.tolist()
                    }
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
