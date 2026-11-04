<<<<<<< SEARCH
    # Final high-precision polish pass with multiple strategies
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        # First: Nelder-Mead with very tight tolerance
        try:
            res_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
        
        # Second: L-BFGS-B polish from the best result
        if time.time() - start_time < DEADLINE - 8:
            try:
                res_polish = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxfun': 500
                    }
                )
                
                if is_valid(res_polish.x):
                    L_val = get_L(res_polish.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_polish.x.copy()
                        
            except Exception:
            pass
=======
    # Simulated annealing restarts to escape local minima
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        sa_start = time.time()
        sa_budget = min(40.0, DEADLINE - 20 - (sa_start - start_time))
        current_best = best_params.copy()
        current_best_L = best_L
        
        # Annealing parameters
        T0 = 0.5
        T_end = 0.001
        n_iter = int(min(300, sa_budget / 0.15))  # ~0.15s per iteration estimate
        
        for it in range(n_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Generate perturbation
            scale = T0 * (T_end / T0) ** (it / max(n_iter - 1, 1))
            perturbation = np.random.randn(3*N + 1) * scale
            candidate = current_best + perturbation
            
            # Ensure angles are reasonable
            candidate[3*N] = candidate[3*N] % 360.0
            
            # Evaluate
            if is_valid(candidate):
                L_cand = get_L(candidate)
                if L_cand < current_best_L:
                    current_best = candidate.copy()
                    current_best_L = L_cand
                    if L_cand < best_L:
                        best_L = L_cand
                        best_params = candidate.copy()
                else:
                    # Metropolis acceptance
                    delta = L_cand - current_best_L
                    if np.random.rand() < np.exp(-delta / max(scale, 1e-6)):
                        current_best = candidate.copy()
                        current_best_L = L_cand
    
    # Final high-precision polish pass with multiple strategies
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        # First: Nelder-Mead with very tight tolerance
        try:
            res_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
        
        # Second: L-BFGS-B polish from the best result
        if time.time() - start_time < DEADLINE - 5:
            try:
                res_polish = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-11,
                        'gtol': 1e-9,
                        'maxfun': 1000
                    }
                )
                
                if is_valid(res_polish.x):
                    L_val = get_L(res_polish.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_polish.x.copy()
                        
            except Exception:
                pass
>>>>>>> REPLACE

