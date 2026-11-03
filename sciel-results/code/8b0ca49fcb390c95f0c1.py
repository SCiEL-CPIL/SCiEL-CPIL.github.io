<<<<<<< SEARCH
def _try_rank(R, n, m, p, deadline, rng, U54_arr, V54_arr, W54_arr):
    """Try to find a rank-R decomposition using numerical optimization."""
    U_shape = (n*m, R)
    V_shape = (m*p, R)
    W_shape = (p*n, R)
    ntriples = n * m * p
    
    # Precompute index arrays for fast residual computation
    i_idx, l_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.flatten()
    l_flat = l_idx.flatten()
    k_flat = k_idx.flatten()
    U_row_idx = i_flat * m + l_flat
    V_row_idx = l_flat * p + k_flat
    W_row_idx = k_flat * n + i_flat
    
    def residuals(x):
        U = x[:n*m*R].reshape(U_shape)
        V = x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
        W = x[n*m*R + m*p*R:].reshape(W_shape)
        
        U_sel = U[U_row_idx, :]
        V_sel = V[V_row_idx, :]
        W_sel = W[W_row_idx, :]
        return (U_sel * V_sel * W_sel).sum(axis=1) - 1.0
    
    rank54 = 54
    if R <= rank54:
        U_init = U54_arr[:, :R].copy()
        V_init = V54_arr[:, :R].copy()
        W_init = W54_arr[:, :R].copy()
    else:
        U_init = np.zeros(U_shape)
        V_init = np.zeros(V_shape)
        W_init = np.zeros(W_shape)
        U_init[:, :rank54] = U54_arr
        V_init[:, :rank54] = V54_arr
        W_init[:, :rank54] = W54_arr
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try multiple strategies
    for attempt in range(8):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Small perturbation of Strassen init
            x_try = x0 + rng.normal(0, 0.05, size=x0.shape)
        elif attempt == 1:
            # Larger perturbation
            x_try = x0 + rng.normal(0, 0.5, size=x0.shape)
        elif attempt == 2:
            # Pure random
            x_try = rng.normal(0, 1, size=x0.shape)
        else:
            # Mix of random and Strassen
            alpha = 0.1 * (attempt - 2)
            x_try = alpha * x0 + (1 - alpha) * rng.normal(0, 1, size=x0.shape)
        
        try:
            result = least_squares(residuals, x_try, method='lm', max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-9:
                U_sol = result.x[:n*m*R].reshape(U_shape)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
                W_sol = result.x[n*m*R + m*p*R:].reshape(W_shape)
                
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                    return U_sol, V_sol, W_sol
        except Exception:
            continue
    
    return None
=======
def _try_rank(R, n, m, p, deadline, rng, U54_arr, V54_arr, W54_arr):
    """Try to find a rank-R decomposition using numerical optimization with structured init."""
    U_shape = (n*m, R)
    V_shape = (m*p, R)
    W_shape = (p*n, R)
    ntriples = n * m * p
    
    # Precompute index arrays for fast residual computation
    i_idx, l_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.flatten()
    l_flat = l_idx.flatten()
    k_flat = k_idx.flatten()
    U_row_idx = i_flat * m + l_flat
    V_row_idx = l_flat * p + k_flat
    W_row_idx = k_flat * n + i_flat
    
    def residuals(x):
        U = x[:n*m*R].reshape(U_shape)
        V = x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
        W = x[n*m*R + m*p*R:].reshape(W_shape)
        
        U_sel = U[U_row_idx, :]
        V_sel = V[V_row_idx, :]
        W_sel = W[W_row_idx, :]
        return (U_sel * V_sel * W_sel).sum(axis=1) - 1.0
    
    rank54 = 54
    if R <= rank54:
        U_init = U54_arr[:, :R].copy()
        V_init = V54_arr[:, :R].copy()
        W_init = W54_arr[:, :R].copy()
    else:
        U_init = np.zeros(U_shape)
        V_init = np.zeros(V_shape)
        W_init = np.zeros(W_shape)
        U_init[:, :rank54] = U54_arr
        V_init[:, :rank54] = V54_arr
        W_init[:, :rank54] = W54_arr
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try multiple strategies with two-stage optimization
    for attempt in range(12):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Small perturbation of Strassen init
            x_try = x0 + rng.normal(0, 0.03, size=x0.shape)
        elif attempt == 1:
            # Medium perturbation
            x_try = x0 + rng.normal(0, 0.2, size=x0.shape)
        elif attempt == 2:
            # Larger perturbation
            x_try = x0 + rng.normal(0, 0.5, size=x0.shape)
        elif attempt == 3:
            # Pure random
            x_try = rng.normal(0, 1, size=x0.shape)
        elif attempt == 4:
            # Scale-free random init
            x_try = rng.uniform(-1, 1, size=x0.shape)
        elif attempt == 5:
            # Binary-like random
            x_try = rng.choice([-1, 0, 1], size=x0.shape)
        elif attempt == 6:
            # Mix 50/50
            x_try = 0.5 * x0 + 0.5 * rng.normal(0, 1, size=x0.shape)
        elif attempt == 7:
            # Mix 80/20
            x_try = 0.8 * x0 + 0.2 * rng.normal(0, 1, size=x0.shape)
        elif attempt == 8:
            # Mix 20/80
            x_try = 0.2 * x0 + 0.8 * rng.normal(0, 1, size=x0.shape)
        elif attempt == 9:
            # Sparse random
            mask = rng.random(size=x0.shape) < 0.3
            x_try = np.where(mask, rng.normal(0, 1, size=x0.shape), 0)
        elif attempt == 10:
            # Very small perturbation
            x_try = x0 + rng.normal(0, 0.01, size=x0.shape)
        else:
            # Alternating pattern
            x_try = x0 + rng.normal(0, 0.1, size=x0.shape) * (1 + 0.5 * np.sin(np.arange(x0.shape)))
        
        try:
            # Stage 1: Coarse optimization with looser tolerance
            result1 = least_squares(residuals, x_try, method='lm', max_nfev=2000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
            
            if result1.success or np.max(np.abs(result1.fun)) < 1e-6:
                # Stage 2: Fine optimization with tighter tolerance
                result2 = least_squares(residuals, result1.x, method='lm', max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result2.success or np.max(np.abs(result2.fun)) < 1e-10:
                    U_sol = result2.x[:n*m*R].reshape(U_shape)
                    V_sol = result2.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
                    W_sol = result2.x[n*m*R + m*p*R:].reshape(W_shape)
                    
                    if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                        return U_sol, V_sol, W_sol
            elif np.max(np.abs(result1.fun)) < 1e-4:
                # Try to polish even if not fully converged
                result2 = least_squares(residuals, result1.x, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if result2.success or np.max(np.abs(result2.fun)) < 1e-10:
                    U_sol = result2.x[:n*m*R].reshape(U_shape)
                    V_sol = result2.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
                    W_sol = result2.x[n*m*R + m*p*R:].reshape(W_shape)
                    
                    if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                        return U_sol, V_sol, W_sol
        except Exception:
            continue
    
    return None
>>>>>>> REPLACE

