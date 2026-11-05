<<<<<<< SEARCH
    # Refine the top candidates
    # We have limited time, so we pick the best 3-5 initial guesses to refine
    num_to_refine = min(5, len(scored_candidates))
    
    for i in range(num_to_refine):
        if time.time() > deadline:
            break
            
        init_val, init_mats, init_ths = scored_candidates[i]
        
        # If this initial guess is already better than our best, update
        if init_val < best_val:
            best_val = init_val
            best_mats = init_mats
            best_ths = init_ths
            
        # Run Nelder-Mead on this candidate
        materials = list(init_mats)
        lo = np.zeros(len(materials))
        hi = np.full(len(materials), max_layer)
        x0 = np.array(init_ths, dtype=float)
        
        # Define objective for this specific material set
        def obj(x):
            return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
            
        x_best, f_best = _nelder_mead(obj, x0, (lo, hi), deadline, max_iter=500)
        
        # Final validity check and constraint enforcement
        x_best = np.clip(x_best, 0.0, max_layer)
        if sum(x_best) > max_total + 1e-6:
            f = max_total / sum(x_best)
            x_best = x_best * f
        x_best = np.clip(x_best, 0.0, max_layer)
        
        val = _mean_reflectance_percent(materials, list(x_best), problem, wls, n_mat)
        
        if val < best_val:
            best_val = val
            best_mats = materials
            best_ths = list(x_best)
=======
    # Refine the top candidates using DE + SA
    num_to_refine = min(5, len(scored_candidates))
    
    for i in range(num_to_refine):
        if time.time() > deadline:
            break
            
        init_val, init_mats, init_ths = scored_candidates[i]
        
        # If this initial guess is already better than our best, update
        if init_val < best_val:
            best_val = init_val
            best_mats = init_mats
            best_ths = init_ths
            
        # Run Differential Evolution on this candidate
        materials = list(init_mats)
        lo = np.zeros(len(materials))
        hi = np.full(len(materials), max_layer)
        x0 = np.array(init_ths, dtype=float)
        
        # Define objective for this specific material set
        def obj(x):
            return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
        
        # Phase 1: DE for global search
        x_best, f_best = _differential_evolution(obj, x0, (lo, hi), deadline, pop_size=24, max_iter=150)
        
        # Phase 2: SA polish for local refinement
        x_best, f_best = _simulated_annealing(obj, x_best, (lo, hi), deadline, max_iter=300)
        
        # Final validity check and constraint enforcement
        x_best = np.clip(x_best, 0.0, max_layer)
        if sum(x_best) > max_total + 1e-6:
            f = max_total / sum(x_best)
            x_best = x_best * f
        x_best = np.clip(x_best, 0.0, max_layer)
        
        val = _mean_reflectance_percent(materials, list(x_best), problem, wls, n_mat)
        
        if val < best_val:
            best_val = val
            best_mats = materials
            best_ths = list(x_best)
>>>>>>> REPLACE

