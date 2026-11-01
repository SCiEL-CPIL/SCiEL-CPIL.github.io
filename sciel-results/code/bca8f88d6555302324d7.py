<<<<<<< SEARCH
def construct():
    """
    Direct approach: use the analytic structure to find optimal coefficients.
    
    The problem is to minimize sqrt(vf * vF) over (a0, a4, a8) with a12 determined.
    
    Strategy:
    1. Use multiple starting points based on analytic insights
    2. L-BFGS-B with exact gradient
    3. Coarse grid + refinement
    
    Key insight: The optimal solution likely has a specific ratio structure.
    The seed (-1.13, 0.04, 0.000308642) gives score 1.0.
    """
    from scipy.optimize import minimize
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Starting points: seed + structured exploration
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        # Add more diverse starting points
        np.array([-1.3, 0.035, 0.00035]),
        np.array([-0.9, 0.055, 0.0006]),
        np.array([-1.7, 0.025, 0.00025]),
        np.array([-0.7, 0.07, 0.0009]),
        np.array([-2.2, 0.018, 0.00018]),
        np.array([-1.4, 0.032, 0.00032]),
        np.array([-0.6, 0.085, 0.0012]),
        np.array([-2.8, 0.012, 0.00012]),
        np.array([-1.18, 0.041, 0.00031]),
        np.array([-1.08, 0.039, 0.00029]),
    ]
    
    # Add random starts with better distribution
    rng = np.random.default_rng(42)
    for _ in range(40):
        # Use log-uniform distribution for better exploration
        a0 = -np.exp(rng.uniform(0.5, 2.0))
        a4 = np.exp(rng.uniform(-3.0, 0.0)) * 0.1
        a8 = np.exp(rng.uniform(-5.0, -1.0)) * 0.001
        starts.append(np.array([a0, a4, a8]))
    
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Coarse grid refinement around best found
    if time.time() < _DEADLINE - 20:
        center = best_params.copy()
        scales = [0.3, 0.1, 0.03, 0.01, 0.003]
        for scale in scales:
            if time.time() > _DEADLINE - 10:
                break
            # Try perturbations in each direction
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 10:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try:
                        val = _bound(trial)
                        if val < best_val:
                            best_val = val
                            best_params = trial.copy()
                            # Refine from here
                            res = minimize(_objective_grad, trial, method='L-BFGS-B',
                                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
                            if res.fun < best_val and np.isfinite(res.fun):
                                best_val = res.fun
                                best_params = res.x.copy()
                    except Exception:
                        pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish with tighter tolerances
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200000, 'xatol': 1e-20, 'fatol': 1e-22, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final exact local solve using Newton's method if gradient is available
    if time.time() < _DEADLINE - 1:
        try:
            from scipy.optimize import newton
            # Try to find zero of gradient components
            for i in range(3):
                def grad_component(x, i=i):
                    _, g = _objective_grad(x)
                    return g[i]
                try:
                    x_new = newton(grad_component, best_params[i], tol=1e-18, maxiter=100)
                    if np.isfinite(x_new):
                        best_params[i] = x_new
                except:
                    pass
            # Evaluate final value
            final_val = _bound(best_params)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return np.array(best_params)
=======
def construct():
    """
    Direct approach: use the analytic structure to find optimal coefficients.
    
    The problem is to minimize sqrt(vf * vF) over (a0, a4, a8) with a12 determined.
    
    Strategy:
    1. Use multiple starting points based on analytic insights
    2. L-BFGS-B with exact gradient (vectorized)
    3. Nelder-Mead derivative-free polish
    4. Final high-precision polish
    """
    from scipy.optimize import minimize
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Starting points: seed + structured exploration
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.3, 0.035, 0.00035]),
        np.array([-0.9, 0.055, 0.0006]),
        np.array([-1.7, 0.025, 0.00025]),
        np.array([-0.7, 0.07, 0.0009]),
        np.array([-2.2, 0.018, 0.00018]),
        np.array([-1.4, 0.032, 0.00032]),
        np.array([-0.6, 0.085, 0.0012]),
        np.array([-2.8, 0.012, 0.00012]),
        np.array([-1.18, 0.041, 0.00031]),
        np.array([-1.08, 0.039, 0.00029]),
    ]
    
    # Add random starts with better distribution
    rng = np.random.default_rng(42)
    for _ in range(60):
        a0 = -np.exp(rng.uniform(0.5, 2.5))
        a4 = np.exp(rng.uniform(-3.5, 0.5)) * 0.1
        a8 = np.exp(rng.uniform(-5.5, -0.5)) * 0.001
        starts.append(np.array([a0, a4, a8]))
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead derivative-free from best and nearby points
    if time.time() < _DEADLINE - 20:
        nm_starts = [best_params.copy()]
        # Add perturbed versions
        for scale in [0.1, 0.05, 0.01]:
            for axis in range(3):
                for direction in [-1, 1]:
                    trial = best_params.copy()
                    trial[axis] += direction * scale
                    nm_starts.append(trial)
        
        for start in nm_starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100000, 'ftol': 1e-25, 'gtol': 1e-22, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead with very tight tolerances
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-22, 'fatol': 1e-25, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
>>>>>>> REPLACE
