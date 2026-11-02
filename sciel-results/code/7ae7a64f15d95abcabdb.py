<<<<<<< SEARCH
def _polish(c, max_time=15.0):
    """Final polish using alternating coordinate updates and local optimization."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Try multiple small perturbations and optimize
    np.random.seed(42)
    for _ in range(5):
        if time.time() - start_time > max_time - 3:
            break
        
        # Small random perturbation
        c_pert = best_c + np.random.normal(0, 0.001, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        try:
            result = minimize(
                _objective_exact,
                c_pert.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Final Delaunay polish
    if time.time() - start_time < max_time - 2:
        try:
            c_delaunay, r_delaunay = _delaunay_optimize(best_c, max_time=min(5.0, max_time - time.time() - 1))
            if r_delaunay > best_r:
                best_c = c_delaunay
                best_r = r_delaunay
        except Exception:
            pass
    
    return best_c, best_r
=======
def _polish(c, max_time=15.0):
    """Final polish using multiple strategies."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Strategy 1: Small perturbations with L-BFGS-B
    np.random.seed(42)
    for _ in range(3):
        if time.time() - start_time > max_time - 4:
            break
        
        # Small random perturbation
        c_pert = best_c + np.random.normal(0, 0.0005, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        try:
            result = minimize(
                _objective_exact,
                c_pert.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Strategy 2: Delaunay polish
    if time.time() - start_time < max_time - 3:
        try:
            c_delaunay, r_delaunay = _delaunay_optimize(best_c, max_time=min(4.0, max_time - time.time() - 2))
            if r_delaunay > best_r:
                best_c = c_delaunay
                best_r = r_delaunay
        except Exception:
            pass
    
    # Strategy 3: Final exact L-BFGS-B
    if time.time() - start_time < max_time - 1:
        try:
            result = minimize(
                _objective_exact,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r
>>>>>>> REPLACE

