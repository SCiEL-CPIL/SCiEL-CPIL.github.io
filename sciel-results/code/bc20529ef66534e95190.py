    # Try to find rank 30
    target_rank = 30
    found_solution = False
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        total_size = size_u + size_v + size_w
        
        # Random initialization
        np.random.seed(42 + restart)
        x0 = np.random.uniform(-1, 1, total_size)
        
        def fun(x):
            err, grad = compute_error_and_grad(x, n, m, p, target_rank)
            return err, grad
            
        try:
            res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if time.time() > deadline:
                break
                
            final_err = res.fun
            if final_err < 1e-6:
                best_rank = target_rank
                best_params = res.x
                found_solution = True
                break
        except Exception:
            pass
            
    if found_solution:
        # ... snap and return logic ...
