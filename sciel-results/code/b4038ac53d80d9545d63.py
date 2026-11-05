<<<<<<< SEARCH
    # Run optimization
    try:
        result = differential_evolution(
            _objective,
            bounds=[(-5, 5)] * (2*stages-2),
            args=(A, B, times, stages),
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            callback=lambda xk, convergence: None
        )
        
        a_opt, b_opt = _enforce_constraints(result.x, stages)
        
        # Verify bounds
        if np.all(a_opt >= -50) and np.all(a_opt <= 50) and np.all(b_opt >= -50) and np.all(b_opt <= 50):
            err_opt = _spectral_norm_error(a_opt, b_opt, A, B, times)
            if err_opt < coefficients._best_err:
                coefficients._best_err = err_opt
                coefficients._best_a = a_opt
                coefficients._best_b = b_opt
            return list(a_opt), list(b_opt)
        else:
            # Clamp
            a_opt = np.clip(a_opt, -50, 50)
            b_opt = np.clip(b_opt, -50, 50)
            # Re-normalize
            a_opt = a_opt / np.sum(a_opt)
            b_opt = b_opt / np.sum(b_opt)
            err_opt = _spectral_norm_error(a_opt, b_opt, A, B, times)
            if err_opt < coefficients._best_err:
                coefficients._best_err = err_opt
                coefficients._best_a = a_opt
                coefficients._best_b = b_opt
            return list(a_opt), list(b_opt)
            
    except Exception as e:
        # Fallback to initial guess
        return list(a_init), list(b_init)
    
    # Deadline check
    if time.time() > _DEADLINE:
        if coefficients._best_a is not None:
            return list(coefficients._best_a), list(coefficients._best_b)
        return list(a_init), list(b_init)
    
    return list(a_init), list(b_init)
=======
    # Simple local search optimization
    # Start with equal weights
    best_a = np.array(a_init)
    best_b = np.array(b_init)
    best_err = _spectral_norm_error(best_a, best_b, A, B, times)
    
    # Store best
    coefficients._best_err = best_err
    coefficients._best_a = best_a
    coefficients._best_b = best_b
    
    # Perform random restarts / local search
    # Budget: ~100s for optimization
    opt_start = time.time()
    max_time = 100.0
    step_size = 0.1
    rng = np.random.default_rng(42)
    
    current_a = best_a.copy()
    current_b = best_b.copy()
    current_err = best_err
    
    # We optimize the free parameters (first stages-1 for each)
    # To keep it simple and robust, we perturb the full vectors and re-normalize
    # Or perturb free params and reconstruct. Reconstructing is safer for constraints.
    
    def get_free(x_full):
        return x_full[:stages-1]
        
    def set_free(x_full, free):
        x_new = x_full.copy()
        x_new[:stages-1] = free
        x_new[-1] = 1.0 - np.sum(free)
        return x_new

    # Initial free params
    free_a = get_free(current_a)
    free_b = get_free(current_b)
    
    iterations = 0
    while time.time() - opt_start < max_time:
        # Random perturbation
        # Scale perturbation by step_size
        # We try to improve by small random steps
        # If we fail several times, reduce step size (simulated annealing-ish)
        # For simplicity, just random steps with acceptance
        
        # Perturb free params
        noise_a = rng.normal(0, step_size, size=stages-1)
        noise_b = rng.normal(0, step_size, size=stages-1)
        
        new_free_a = free_a + noise_a
        new_free_b = free_b + noise_b
        
        # Construct full vectors
        new_a = set_free(current_a, new_free_a)
        new_b = set_free(current_b, new_free_b)
        
        # Check bounds
        if np.any(new_a < -50) or np.any(new_a > 50) or np.any(new_b < -50) or np.any(new_b > 50):
            continue
            
        # Evaluate
        new_err = _spectral_norm_error(new_a, new_b, A, B, times)
        
        if new_err < current_err:
            current_a = new_a
            current_b = new_b
            current_err = new_err
            free_a = new_free_a
            free_b = new_free_b
            
            # Update global best
            if current_err < coefficients._best_err:
                coefficients._best_err = current_err
                coefficients._best_a = current_a.copy()
                coefficients._best_b = current_b.copy()
                
            # Increase step size slightly on success? Or keep.
            # Let's keep step size constant for simplicity, or decrease slowly.
            # Actually, random walk is slow. Let's try coordinate ascent?
            # No, random walk is safer to implement quickly.
            
        else:
            # Occasionally accept worse solutions to escape local minima?
            # Simple hill climbing: reject.
            # To prevent getting stuck, we can reduce step size if no improvement for a while.
            pass
        iterations += 1
        
        # If we are getting close to deadline, break
        if time.time() - opt_start > max_time - 5.0:
            break
            
    # Return the best found
    if coefficients._best_a is not None:
        return list(coefficients._best_a), list(coefficients._best_b)
    return list(a_init), list(b_init)
>>>>>>> REPLACE
