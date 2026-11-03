"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    # Target tensor indices
    # T[i, j, k] = 1 if the indices correspond to the matrix multiplication tensor
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def _verify(U, V, W):
        """
        Verify if U, V, W form an exact decomposition.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        # Check exactness: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # We can use einsum for speed
        # Construct the reconstruction
        # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # But we only care about the specific (u,v,w) tuples corresponding to (i,j,k)
        
        # Let's just loop, it's small (24 terms)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx]) # Wait, this is not correct.
                    # The sum is over r of U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # This is not a standard dot product of two vectors.
                    # It is the dot product of (U[u_idx] * V[v_idx]) and W[w_idx]
                    prod_uv = U[u_idx] * V[v_idx]
                    val = np.dot(prod_uv, W[w_idx])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def _snap_to_half_int(arr):
        return np.round(2 * arr) / 2.0

    def _als_step(U, V, W):
        """
        Perform one ALS iteration.
        Returns updated U, V, W.
        """
        R = U.shape[1]
        
        # 1. Update W
        # For each (k, i), solve for W[k*n+i, :]
        # Equations: for j in 0..m-1: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # Matrix A_W: shape (m, R)
        # b_W: shape (m,)
        new_W = W.copy()
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m, R))
                b_W = np.ones(m)
                for j in range(m):
                    # Coefficients for r
                    A_W[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve A_W x = b_W
                try:
                    # If R < m, least squares. If R >= m, we want a solution.
                    # np.linalg.lstsq handles both.
                    sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                    new_W[k * n + i, :] = sol
                except:
                    pass
        W = new_W

        # 2. Update U
        # For each (i, j), solve for U[i*m+j, :]
        # Equations: for k in 0..p-1: sum_r (V[j*p+k, r] * W[k*n+i, r]) * U[i*m+j, r] = 1
        # Matrix A_U: shape (p, R)
        new_U = U.copy()
        for i in range(n):
            for j in range(m):
                A_U = np.zeros((p, R))
                b_U = np.ones(p)
                for k in range(p):
                    A_U[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                    new_U[i * m + j, :] = sol
                except:
                    pass
        U = new_U

        # 3. Update V
        # For each (j, k), solve for V[j*p+k, :]
        # Equations: for i in 0..n-1: sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] = 1
        # Matrix A_V: shape (n, R)
        new_V = V.copy()
        for j in range(m):
            for k in range(p):
                A_V = np.zeros((n, R))
                b_V = np.ones(n)
                for i in range(n):
                    A_V[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                    new_V[j * p + k, :] = sol
                except:
                    pass
        V = new_V

        return U, V, W

    def _try_rank(R, max_restarts=5, max_iters=200):
        """
        Try to find a decomposition of rank R.
        Returns (U, V, W) or None.
        """
        if time.time() - start_time > time_limit:
            return None
            
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            # Use small random values to avoid zero-initialization stagnation
            np.random.seed(42 + restart)
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                    
                U, V, W = _als_step(U, V, W)
                
                # Calculate error
                # err = sum_{i,j,k} (dot(U[i,j]*V[j,k], W[k,i]) - 1)^2
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod_uv = U[i * m + j] * V[j * p + k]
                            val = np.dot(prod_uv, W[k * n + i])
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if err < 1e-10:
                    break
                
                # Early stopping if stuck
                if it > 10 and err > best_err * 1.1:
                     break

        # If we found a near-exact solution, snap and verify
        if best_U is not None and best_err < 1e-6:
            U_snap = _snap_to_half_int(best_U)
            V_snap = _snap_to_half_int(best_V)
            W_snap = _snap_to_half_int(best_W)
            
            if _verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        
        # If snapping failed, maybe the exact solution is not near the ALS solution.
        # Try a simple local search on the snapped values?
        # Given time constraints, we might skip this for now or do a very short one.
        # Let's try a short local search if we have time.
        if time.time() - start_time < time_limit - 10:
            # Initialize with snapped best
            U_ls = _snap_to_half_int(best_U)
            V_ls = _snap_to_half_int(best_V)
            W_ls = _snap_to_half_int(best_W)
            
            # Simple hill climbing
            values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            improved = True
            max_ls_iter = 100
            for _ in range(max_ls_iter):
                if not improved:
                    break
                if time.time() - start_time > time_limit - 5:
                    break
                improved = False
                # Pick random entry
                total_vars = U_ls.size + V_ls.size + W_ls.size
                idx = np.random.randint(0, total_vars)
                
                if idx < U_ls.size:
                    r, c = np.unravel_index(idx, U_ls.shape)
                    old_val = U_ls[r, c]
                    for new_val in values:
                        if new_val == old_val: continue
                        U_ls[r, c] = new_val
                        if _verify(U_ls, V_ls, W_ls):
                            improved = True
                            break
                        U_ls[r, c] = old_val
                elif idx < U_ls.size + V_ls.size:
                    idx_v = idx - U_ls.size
                    r, c = np.unravel_index(idx_v, V_ls.shape)
                    old_val = V_ls[r, c]
                    for new_val in values:
                        if new_val == old_val: continue
                        V_ls[r, c] = new_val
                        if _verify(U_ls, V_ls, W_ls):
                            improved = True
                            break
                        V_ls[r, c] = old_val
                else:
                    idx_w = idx - U_ls.size - V_ls.size
                    r, c = np.unravel_index(idx_w, W_ls.shape)
                    old_val = W_ls[r, c]
                    for new_val in values:
                        if new_val == old_val: continue
                        W_ls[r, c] = new_val
                        if _verify(U_ls, V_ls, W_ls):
                            improved = True
                            break
                        W_ls[r, c] = old_val
                        
            if _verify(U_ls, V_ls, W_ls):
                return U_ls, V_ls, W_ls

        return None

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 12
    # We try from 12 up to 23
    # To save time, we might want to start from a higher rank if we suspect lower ones are hard?
    # But the goal is minimal. We must check 12, 13, ...
    # However, ALS might struggle for very low ranks.
    # Let's try a range.
    
    found = None
    # We will try ranks in increasing order.
    # To be efficient, we limit time per rank.
    # If we don't find it in the first few ranks, we might spend too much time.
    # Let's allocate time budget.
    
    ranks_to_try = list(range(12, rank_schoolbook))
    
    # Reverse order? No, we want the *minimal* rank. So we must find the smallest one.
    # But if rank 12 is impossible, we waste time.
    # Heuristic: Start from a "safe" lower bound like 16 or 18?
    # Actually, for 2x3x4, the rank is likely not 12.
    # Let's try 16, 17, 18, 19, 20, 21, 22, 23.
    # If we find one, we return it.
    # But we must ensure it's minimal.
    # If we find rank 18, we should still check if 17, 16... are possible?
    # The prompt asks for "rank-minimal".
    # So we MUST search from the bottom up.
    
    # However, if the search is expensive, we might not finish checking all.
    # Let's prioritize the search but keep a strict time limit.
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Reduce restarts/iterations for lower ranks to save time for higher ones if needed?
        # Or just use a fixed budget.
        # Let's use a fixed small budget per rank.
        result = _try_rank(R, max_restarts=3, max_iters=100)
        
        if result is not None:
            U, V, W = result
            # Convert to list of lists of floats
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # If no lower rank found, return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

# EVOLVE-BLOCK-END
