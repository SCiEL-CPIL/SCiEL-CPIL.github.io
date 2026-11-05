<<<<<<< SEARCH
def _optimize():
    """Run optimization to find best coefficients."""
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    stages = 12
    
    # Parameterization: use 22 free parameters + 2 determined by constraints
    # a = a_free + (1 - sum(a_free)) / stages * ones  -> this doesn't work directly
    # Better: a[12] = 1 - sum(a[0:11]), b[12] = 1 - sum(b[0:11])
    
    def constrained_objective(x):
        a = x[:11].copy()
        b = x[11:22].copy()
        a = np.append(a, 1.0 - np.sum(a))
        b = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        
        return _objective(np.concatenate([a, b]), A, B, times)
    
    # Initial guess: Strang-like
    x0 = np.zeros(22)
    for k in range(11):
        x0[k] = 1.0 / 12
        x0[k + 11] = 1.0 / 12
    
    # Use SLSQP with bounds
    bounds = [(-50, 50)] * 22
    
    from scipy.optimize import minimize
    result = minimize(constrained_objective, x0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-12})
    
    a = result.x[:11].copy()
    b = result.x[11:22].copy()
    a = np.append(a, 1.0 - np.sum(a))
    b = np.append(b, 1.0 - np.sum(b))
    
    # Clip to valid range
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize
    a = a / np.sum(a) if abs(np.sum(a)) > 1e-15 else np.ones(12) / 12
    b = b / np.sum(b) if abs(np.sum(b)) > 1e-15 else np.ones(12) / 12
    
    return a.tolist(), b.tolist()
=======
def _optimize():
    """Run optimization to find best coefficients."""
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    stages = 12
    
    def constrained_objective(x):
        a = x[:11].copy()
        b = x[11:22].copy()
        a = np.append(a, 1.0 - np.sum(a))
        b = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        
        return _objective(np.concatenate([a, b]), A, B, times)
    
    # Initial guess: Uniform (1/12)
    x0 = np.ones(22) / 12.0
    
    # Run Nelder-Mead
    # We don't strictly enforce bounds in NM, but the objective penalizes them.
    # To be safer, we can clip x inside the objective if needed, but penalty is usually enough.
    result = _nelder_mead(constrained_objective, x0, bounds=None, max_iter=100)
    
    a = result[:11].copy()
    b = result[11:22].copy()
    a = np.append(a, 1.0 - np.sum(a))
    b = np.append(b, 1.0 - np.sum(b))
    
    # Clip to valid range
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize to ensure sum is exactly 1
    a = a / np.sum(a) if abs(np.sum(a)) > 1e-15 else np.ones(12) / 12
    b = b / np.sum(b) if abs(np.sum(b)) > 1e-15 else np.ones(12) / 12
    
    return a.tolist(), b.tolist()
>>>>>>> REPLACE

