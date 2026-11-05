def _try_rank26_warm_start():
    n, m, p = 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    target_rank = 26
    
    U28, V28, W28 = _build_rank28()
    U28_arr = np.array([[float(x) for x in row] for row in U28], dtype=float)
    V28_arr = np.array([[float(x) for x in row] for row in V28], dtype=float)
    W28_arr = np.array([[float(x) for x in row] for row in W28], dtype=float)
    
    best_obj = np.inf
    
    # Try random pairs of columns to remove
    np.random.seed(42)
    trials = 20
    for _ in range(trials):
        cols_to_remove = np.random.choice(28, 2, replace=False)
        
        U_init = np.delete(U28_arr, cols_to_remove, axis=1)
        V_init = np.delete(V28_arr, cols_to_remove, axis=1)
        W_init = np.delete(W28_arr, cols_to_remove, axis=1)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def _objective(params):
            U = params[:target_rank*nm].reshape(target_rank, nm).T
            V = params[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
            W = params[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
            
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        residual += (s - 1.0)**2
            return residual
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                
                if result.fun < 1e-8:
                    U = result.x[:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    res_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                                res_rounded += (s - 1.0)**2
                    
                    if res_rounded < 1e-8:
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U_r]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V_r]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W_r]
                        
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
        except:
            continue
            
    return None
