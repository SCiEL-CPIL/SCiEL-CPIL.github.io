    def search_decomp(n, m, p, min_rank, max_rank, time_limit):
        # Returns (U, V, W, rank) or None
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        best_rank = None
        best_U = None
        best_V = None
        best_W = None
        
        deadline = time.time() + time_limit
        
        for R in range(min_rank, max_rank + 1):
            if time.time() > deadline:
                break
            
            # Try a few random initializations
            num_trials = 3
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + R * 100 + trial)
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
                
                def objective_and_grad(params):
                    size_U = n * m * R
                    size_V = m * p * R
                    U_flat = params[0:size_U]
                    V_flat = params[size_U:size_U+size_V]
                    W_flat = params[size_U+size_V:]
                    
                    U_mat = U_flat.reshape(n*m, R)
                    V_mat = V_flat.reshape(m*p, R)
                    W_mat = W_flat.reshape(p*n, R)
                    
                    T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                    diff = T - T_comp
                    loss = np.sum(diff * diff)
                    
                    grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
                    grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
                    grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
                    
                    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                    return loss, grad
                
                from scipy.optimize import minimize
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12})
                    if res.fun < 1e-6:
                        U_opt = res.x[0:n*m*R].reshape(n*m, R)
                        V_opt = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                        W_opt = res.x[n*m*R+m*p*R:].reshape(p*n, R)
                        
                        # Verify
                        T_comp = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
                        if np.allclose(T_comp, T, atol=1e-5):
                            return U_opt, V_opt, W_opt, R
                except Exception:
                    continue
        
        return None
    