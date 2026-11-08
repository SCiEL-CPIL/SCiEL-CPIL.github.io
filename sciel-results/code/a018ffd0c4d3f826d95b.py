    # Optimization using Nelder-Mead
    def objective(params):
        p = parametric_pulse(params)
        return compute_score(p)
    
    # Initial guess: sigma=6, alpha=0.5, area_scale=1.0
    x0 = np.array([6.0, 0.5, 1.0])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                              'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        best_pulse = parametric_pulse(best_params)
        return best_pulse

    # Try a few more local searches around the best point
    best_params_copy = best_params.copy()
    
    for restart in range(50):
        if time.time() - start_time > deadline - 20:
            break
        
        # Perturb parameters
        # Wider perturbation
        perturbation = np.random.normal(0, 1.0, size=3)
        new_params = best_params_copy + perturbation
        
        # Ensure sigma is positive
        new_params[0] = max(new_params[0], 2.0)
        
        new_score = objective(new_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params_copy = new_params

    best_pulse = parametric_pulse(best_params_copy)
    
    # Final validity check: if invalid, scale down to make it valid.
    # This is a fallback. The penalty should have kept it valid.
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(best_pulse[0]) / dt
        else:
            dp = np.linalg.norm(best_pulse[i] - best_pulse[i-1]) / dt
        if dp > max_slew:
            max_slew = dp
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
