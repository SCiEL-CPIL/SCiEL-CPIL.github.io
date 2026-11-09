import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
                
    # Precompute indices for efficiency if needed, but T is small.
    
    def solve_als(T, rank, max_iter=200, tol=1e-12, seed=0):
        """
        Alternating Least Squares for CP decomposition.
        T: (dim_u, dim_v, dim_w)
        rank: R
        Returns: U, V, W if converged to error < tol, else None
        """
        np.random.seed(seed)
        # Initialize with random values. 
        # Scaling matters. We can try to match the scale of T.
        # T has 1s and 0s. 
        # Let's initialize with small random values to start, or random normal.
        # If we use random normal, the initial error might be large.
        # A better init: 
        # U ~ N(0, 1), V ~ N(0, 1), W ~ N(0, 1)
        
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # We can normalize to prevent explosion
        # But ALS is scale invariant in terms of direction, just not magnitude.
        # Actually, ALS steps are:
        # U[i, :] = argmin ...
        
        for it in range(max_iter):
            # Update U
            for i in range(dim_u):
                # We want to minimize || T[i, :, :] - sum_r U[i, r] V[:, r] W[:, r] ||_F^2
                # Reshape T[i, :, :] to (dim_v * dim_w, 1)
                # Construct matrix M where M[(j, k), r] = V[j, r] * W[k, r]
                # M is (dim_v * dim_w, rank)
                M = np.outer(V, W).reshape(dim_v * dim_w, rank)
                # b is T[i, :, :].flatten()
                b = T[i, :, :].flatten()
                # Solve M x = b
                # Use least squares
                try:
                    U[i, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None
                    
            # Update V
            for j in range(dim_v):
                M = np.outer(U, W).reshape(dim_u * dim_w, rank)
                b = T[:, j, :].flatten()
                try:
                    V[j, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None
                    
            # Update W
            for k in range(dim_w):
                M = np.outer(U, V).reshape(dim_u * dim_v, rank)
                b = T[:, :, k].flatten()
                try:
                    W[k, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None
            
            # Check convergence
            # Compute error
            # T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            # err = np.max(np.abs(T_est - T))
            # To save time, we can check every 10 iterations or use a norm.
            if it % 5 == 0 or it == max_iter - 1:
                T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(T_est - T))
                if err < tol:
                    return U, V, W
                    
        # Final check
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T_est - T)) < tol:
            return U, V, W
            
        return None

    # Search for minimal rank
    # Lower bound for 2x3x7 is not 1. 
    # Known bounds: 
    # Strassen for 2x2x2 is 7.
    # For 2x3x7, let's guess it's around 15-20.
    # We'll search from 10 upwards.
    # If we don't find it by 42, we fall back to schoolbook.
    
    for rank in range(10, 43):
        # Try multiple restarts
        for restart in range(20):
            res = solve_als(T, rank, max_iter=100, tol=1e-9, seed=42 + restart * 100 + rank)
            if res is not None:
                U, V, W = res
                # Double check with a tighter tolerance
                T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.max(np.abs(T_est - T)) < 1e-9:
                    # Return as lists of lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
