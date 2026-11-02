<<<<<<< SEARCH
    # If no valid solution found via optimization, fallback to a known valid configuration
    if best_params is None:
=======
    # Final high-precision polish pass if we have a valid solution
    if best_params is not None and time.time() < deadline - 10:
        try:
            x0 = best_params.copy()
            
            res_polish = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_polish.fun < best_s:
                pos = res_polish.x[:2 * N].reshape(N, 2)
                alphas = res_polish.x[2 * N : 2 * N + N]
                s = res_polish.x[2 * N + N]
                theta = res_polish.x[2 * N + N + 1]
                
                penalty = compute_penalty(pos, alphas, s, theta)
                
                if penalty < 1e-6:
                    best_s = s
                    best_params = res_polish.x.copy()
                    
        except Exception:
            pass

    # If no valid solution found via optimization, fallback to a known valid configuration
    if best_params is None:
>>>>>>> REPLACE
