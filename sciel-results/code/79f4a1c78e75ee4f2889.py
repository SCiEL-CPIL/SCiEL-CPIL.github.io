<<<<<<< SEARCH
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def to_half_integer(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    # Try to find lower rank decompositions using numerical methods
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    # Try a few candidate ranks close to schoolbook
    for target_rank in range(104, 94, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        for restart in range(2):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 10 + restart)
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            if has_scipy:
                def flatten(params):
                    U_flat = params[:n*m*target_rank]
                    V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = params[n*m*target_rank + m*p*target_rank:]
                    return U_flat.reshape(n*m, target_rank), \
                           V_flat.reshape(m*p, target_rank), \
                           W_flat.reshape(p*n, target_rank)
                
                def obj(params):
                    U_l, V_l, W_l = flatten(params)
                    total = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(target_rank):
                                    val += U_l[i*m+j, c] * V_l[j*p+k, c] * W_l[k*n+i, c]
                                total += (val - 1.0)**2
                    return total
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    result = minimize(obj, x0, method='L-BFGS-B', 
                                     options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12})
                    if result.fun < 1e-10:
                        U_opt, V_opt, W_opt = flatten(result.x)
                        U_h = to_half_integer(U_opt)
                        V_h = to_half_integer(V_opt)
                        W_h = to_half_integer(W_opt)
                        if verify(U_h, V_h, W_h, n, m, p, tol=1e-6):
                            best_rank = target_rank
                            best_result = {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                            break
                except Exception:
                    pass
        
        if best_rank == target_rank:
            break
    
    return best_result
=======
    return best_result
>>>>>>> REPLACE
