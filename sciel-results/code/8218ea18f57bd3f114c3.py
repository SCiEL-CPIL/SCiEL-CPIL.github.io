"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.
Uses scipy.optimize.least_squares to find a decomposition for a given rank,
then verifies if it can be represented with half-integers.
"""

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, k: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        summed over appropriate indices.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    import numpy as np
    from scipy.optimize import least_squares
    import time
    import random

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    lower_bound = max(n*m, m*p, n*p)
    
    import time
    start_time = time.time()
    max_search_time = 140  # Leave 10s buffer

    def try_rank(rank, n, m, p):
        """
        Attempt to find a decomposition of the given rank using least_squares.
        Returns (U, V, W) if found, else None.
        """
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_vars = size_u + size_v + size_w
        
        def residuals(x):
            u_flat = x[:size_u]
            v_flat = x[size_u:size_u+size_v]
            w_flat = x[size_u+size_v:]
            
            U = u_flat.reshape((n, m, rank))
            V = v_flat.reshape((m, p, rank))
            W = w_flat.reshape((p, n, rank))
            
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # U: (n, m, r)
            # V: (m, p, r)
            # W: (p, n, r) -> indices k, i, r
            T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            return (T - 1.0).flatten()
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > max_search_time:
                return None
            
            # Random initialization
            # Scale 1.0
            x0 = np.random.uniform(-1, 1, total_vars)
            
            try:
                # Use 'lm' for speed, 'trf' if bounds needed (not here)
                sol = least_squares(residuals, x0, method='lm', max_nfev=500)
            except Exception:
                continue
            
            # Check residual
            if np.linalg.norm(sol.fun) < 1e-5:
                u_flat = sol.x[:size_u]
                v_flat = sol.x[size_u:size_u+size_v]
                w_flat = sol.x[size_u+size_v:]
                
                # Round to nearest 0.5
                u_rounded = np.round(u_flat * 2) / 2
                v_rounded = np.round(v_flat * 2) / 2
                w_rounded = np.round(w_flat * 2) / 2
                
                # Reshape to matrices
                U_mat = u_rounded.reshape((n * m, rank))
                V_mat = v_rounded.reshape((m * p, rank))
                W_mat = w_rounded.reshape((p * n, rank))
                
                # Verify exactness
                if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p):
                    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
            
        return None

    # Search ranks from lower bound up
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() - start_time > max_search_time:
            break
            
        result = try_rank(rank, n, m, p)
        if result:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
            
    return best
