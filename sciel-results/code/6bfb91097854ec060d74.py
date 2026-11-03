<<<<<<< SEARCH
def _newton_search(rank, n, m, p, seed, max_iter=50, tol=1e-10):
    """
    Attempt to find a rank-R decomposition using Newton's method with efficient Jacobian.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    rng = random.Random(seed)
    
    # Initialize with small random values
    U = np.random.uniform(-0.5, 0.5, (n*m, rank))
    V = np.random.uniform(-0.5, 0.5, (m*p, rank))
    W = np.random.uniform(-0.5, 0.5, (p*n, rank))
    
    for _ in range(max_iter):
        # Compute residuals
        res = _fast_residuals(U, V, W, n, m, p, rank)
        if np.linalg.norm(res) < tol:
            break
        
        # Compute Jacobian efficiently
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        total_vars = num_u + num_v + num_w
        
        J = np.zeros((n*m*p, total_vars))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    
                    for r in range(rank):
                        # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                        var_idx_u = (i*m+j)*rank + r
                        J[eq_idx, var_idx_u] = V[j*p+k, r] * W[k*n+i, r]
                        
                        # d/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                        var_idx_v = num_u + (j*p+k)*rank + r
                        J[eq_idx, var_idx_v] = U[i*m+j, r] * W[k*n+i, r]
                        
                        # d/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                        var_idx_w = num_u + num_v + (k*n+i)*rank + r
                        J[eq_idx, var_idx_w] = U[i*m+j, r] * V[j*p+k, r]
        
        # Solve J delta = -res.flatten()
        try:
            delta = np.linalg.lstsq(J, -res.flatten(), rcond=None)[0]
        except Exception:
            break
        
        # Update
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        U_flat += delta[:num_u]
        V_flat += delta[num_u:num_u+num_v]
        W_flat += delta[num_u+num_v:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
    
    # Check final residual
    res = _fast_residuals(U, V, W, n, m, p, rank)
    if np.linalg.norm(res) < 1e-6:
        return U, V, W
    return None
=======
def _newton_search(rank, n, m, p, seed, max_iter=80, tol=1e-10):
    """
    Attempt to find a rank-R decomposition using Newton's method with efficient Jacobian.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    rng = random.Random(seed)
    
    # Initialize with structured values based on known patterns
    # Use a mix of structured and random initialization
    if seed % 3 == 0:
        # Structured initialization: try to build from smaller blocks
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try to decompose using a block approach
        # For 2x3x7, we can think of it as combining 2x2x2 and 2x2x7 or similar
        # Initialize with identity-like patterns
        for r in range(min(rank, n*m)):
            if r < n*m:
                U[r, r] = 1.0
        for r in range(min(rank, m*p)):
            if r < m*p:
                V[r, r] = 1.0
        for r in range(min(rank, p*n)):
            if r < p*n:
                W[r, r] = 1.0
        
        # Add some random perturbation to break symmetry
        U += rng.uniform(-0.1, 0.1, U.shape)
        V += rng.uniform(-0.1, 0.1, V.shape)
        W += rng.uniform(-0.1, 0.1, W.shape)
    elif seed % 3 == 1:
        # Random initialization with larger magnitude
        U = rng.uniform(-1.0, 1.0, (n*m, rank))
        V = rng.uniform(-1.0, 1.0, (m*p, rank))
        W = rng.uniform(-1.0, 1.0, (p*n, rank))
    else:
        # Sparse random initialization
        U = rng.uniform(-0.5, 0.5, (n*m, rank))
        V = rng.uniform(-0.5, 0.5, (m*p, rank))
        W = rng.uniform(-0.5, 0.5, (p*n, rank))
        # Zero out 50% of entries
        mask_u = rng.random((n*m, rank)) > 0.5
        mask_v = rng.random((m*p, rank)) > 0.5
        mask_w = rng.random((p*n, rank)) > 0.5
        U[mask_u] = 0
        V[mask_v] = 0
        W[mask_w] = 0
    
    for iteration in range(max_iter):
        # Compute residuals
        res = _fast_residuals(U, V, W, n, m, p, rank)
        res_norm = np.linalg.norm(res)
        if res_norm < tol:
            break
        
        # Compute Jacobian efficiently using vectorized operations
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        total_vars = num_u + num_v + num_w
        
        J = np.zeros((n*m*p, total_vars))
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    
                    # Vectorized Jacobian computation for this equation
                    # d/dU[i,j,r] = V[j,k,r] * W[k,i,r]
                    J[eq_idx, (i*m+j)*rank:(i*m+j)*rank+rank] = \
                        V_reshaped[j, k, :] * W_reshaped[k, i, :]
                    
                    # d/dV[j,k,r] = U[i,j,r] * W[k,i,r]
                    J[eq_idx, num_u + (j*p+k)*rank:num_u + (j*p+k)*rank+rank] = \
                        U_reshaped[i, j, :] * W_reshaped[k, i, :]
                    
                    # d/dW[k,i,r] = U[i,j,r] * V[j,k,r]
                    J[eq_idx, num_u + num_v + (k*n+i)*rank:num_u + num_v + (k*n+i)*rank+rank] = \
                        U_reshaped[i, j, :] * V_reshaped[j, k, :]
        
        # Solve J delta = -res.flatten() using least squares
        try:
            delta = np.linalg.lstsq(J, -res.flatten(), rcond=None)[0]
        except Exception:
            break
        
        # Damping: use a step size that doesn't increase the residual
        step = 1.0
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Try full step first
        U_new = (U_flat + step * delta[:num_u]).reshape(n*m, rank)
        V_new = (V_flat + step * delta[num_u:num_u+num_v]).reshape(m*p, rank)
        W_new = (W_flat + step * delta[num_u+num_v:]).reshape(p*n, rank)
        
        res_new = _fast_residuals(U_new, V_new, W_new, n, m, p, rank)
        res_new_norm = np.linalg.norm(res_new)
        
        # If residual increased, use a smaller step
        if res_new_norm > res_norm:
            for factor in [0.5, 0.25, 0.1, 0.05, 0.01]:
                U_new = (U_flat + factor * delta[:num_u]).reshape(n*m, rank)
                V_new = (V_flat + factor * delta[num_u:num_u+num_v]).reshape(m*p, rank)
                W_new = (W_flat + factor * delta[num_u+num_v:]).reshape(p*n, rank)
                res_new = _fast_residuals(U_new, V_new, W_new, n, m, p, rank)
                res_new_norm = np.linalg.norm(res_new)
                if res_new_norm < res_norm:
                    step = factor
                    break
            else:
                # No improvement, break
                break
        
        U, V, W = U_new, V_new, W_new
    
    # Check final residual
    res = _fast_residuals(U, V, W, n, m, p, rank)
    if np.linalg.norm(res) < 1e-6:
        return U, V, W
    return None
>>>>>>> REPLACE

