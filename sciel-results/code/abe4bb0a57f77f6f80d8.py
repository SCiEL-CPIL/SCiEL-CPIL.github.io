<<<<<<< SEARCH
    T_start = 10.0
    T_end = 0.01
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Cooling
        progress = it / n_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Generate neighbor: perturb 1-2 hexagons
        n_perturb = np.random.randint(1, 3)
        new_params = current_params.copy()
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            new_params[3*idx] += np.random.normal(0, 0.5)
            new_params[3*idx+1] += np.random.normal(0, 0.5)
            new_params[3*idx+2] += np.random.normal(0, 15)
        
        new_val = _objective(new_params, outer_center, 0.0)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val < current_val:
            current_params = new_params
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = current_params.copy()
        else:
            delta = new_val - current_val
            if delta < 100:  # Don't accept huge penalties
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_params = new_params
                    current_val = new_val
    
    # Local polish with Nelder-Mead
    if time.time() - t0 < DEADLINE - 10:
        from scipy.optimize import minimize
        try:
            res = minimize(lambda p: _objective(p, outer_center, 0.0),
                          best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
=======
    T_start = 5.0
    T_end = 0.01
    # Reduce iterations to fit in time
    n_iterations = 1000
    
    for it in range(n_iterations):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # Cooling
        progress = it / n_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Generate neighbor: perturb 1-2 hexagons
        n_perturb = np.random.randint(1, 3)
        new_params = current_params.copy()
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            new_params[3*idx] += np.random.normal(0, 0.3)
            new_params[3*idx+1] += np.random.normal(0, 0.3)
            new_params[3*idx+2] += np.random.normal(0, 10)
        
        new_val = _objective(new_params, outer_center, 0.0)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val < current_val:
            current_params = new_params
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = current_params.copy()
        else:
            delta = new_val - current_val
            if delta < 50:  # Don't accept huge penalties
                prob = math.exp(-delta / T)
                if np.random.random() < prob:
                    current_params = new_params
                    current_val = new_val
    
    # Local polish with Nelder-Mead
    if time.time() - t0 < DEADLINE - 10:
        from scipy.optimize import minimize
        try:
            res = minimize(lambda p: _objective(p, outer_center, 0.0),
                          best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
>>>>>>> REPLACE
