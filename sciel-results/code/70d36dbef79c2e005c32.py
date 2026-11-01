<<<<<<< SEARCH
def _optimize_delaunay(c0, max_time=100.0):
    """
    Optimize using Delaunay-based soft-min.
    """
    start_time = time.time()
    c = c0.copy()
    n = c.shape[0]
    
    # Initial triangulation
    try:
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
    except Exception:
        # Fallback if triangulation fails (e.g., collinear points)
        # Just use all pairs? No, that's expensive.
        # Add small noise and retry
        c += np.random.normal(0, 1e-6, c.shape)
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
        
    beta = 50.0
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Optimization loop with annealing beta
    # We re-compute Delaunay every few steps to keep edges up to date
    step_count = 0
    retri_freq = 10
    
    def _obj_and_jac(params):
        return _soft_min_objective_and_grad(params, beta, tri_edges, boundary_active_mask=None)
        
    # We use L-BFGS-B
    # Note: We need to update 'edges' inside the closure or pass it via a mutable container
    # Since edges change, we can't easily use a static closure for L-BFGS-B if we want to update them during the run.
    # Strategy: Run L-BFGS-B for a short duration, then re-triangulate, then continue.
    
    max_iter_per_chunk = 50
    while time.time() < start_time + max_time - 5:
        # Define functions that capture current 'edges' and 'beta'
        def _f(p):
            return _soft_min_objective_and_grad(p, beta, edges)[0]
            
        def _g(p):
            return _soft_min_objective_and_grad(p, beta, edges)[1]
            
        try:
            result = minimize(
                _f,
                c.flatten(),
                jac=_g,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                options={'maxiter': max_iter_per_chunk, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                c = c_new
            else:
                # If no improvement, accept anyway to move out of local minima?
                # Or keep best? Usually keep best for stability.
                # Let's move to c_new to explore, but track best.
                c = c_new
                
        except Exception as e:
            # If optimization fails, break or continue
            break
        
        # Increase beta to sharpen the soft-min
        beta *= 1.5
        if beta > 500:
            beta = 500
            
        # Re-triangulate periodically
        step_count += 1
        if step_count % retri_freq == 0:
            try:
                # Add tiny noise to avoid degenerate triangulations
                c_perturbed = c + np.random.normal(0, 1e-8, c.shape)
                c_perturbed = np.clip(c_perturbed, -0.5 + 1e-12, 0.5 - 1e-12)
                tri = Delaunay(c_perturbed)
                edges = _get_delaunay_edges(tri)
            except Exception:
                pass
                
    return best_c, best_r
=======
def _optimize_delaunay(c0, max_time=100.0):
    """
    Optimize using Delaunay-based soft-min.
    """
    start_time = time.time()
    c = c0.copy()
    n = c.shape[0]
    
    # Initial triangulation
    try:
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
    except Exception:
        # Fallback if triangulation fails (e.g., collinear points)
        # Just use all pairs? No, that's expensive.
        # Add small noise and retry
        c += np.random.normal(0, 1e-6, c.shape)
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
        
    beta = 30.0
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Optimization loop with annealing beta
    # We re-compute Delaunay every few steps to keep edges up to date
    step_count = 0
    retri_freq = 5
    
    def _obj_and_jac(params):
        return _soft_min_objective_and_grad(params, beta, tri_edges, boundary_active_mask=None)
        
    # We use L-BFGS-B
    # Note: We need to update 'edges' inside the closure or pass it via a mutable container
    # Since edges change, we can't easily use a static closure for L-BFGS-B if we want to update them during the run.
    # Strategy: Run L-BFGS-B for a short duration, then re-triangulate, then continue.
    
    max_iter_per_chunk = 30
    while time.time() < start_time + max_time - 5:
        # Define functions that capture current 'edges' and 'beta'
        def _f(p):
            return _soft_min_objective_and_grad(p, beta, edges)[0]
            
        def _g(p):
            return _soft_min_objective_and_grad(p, beta, edges)[1]
            
        try:
            result = minimize(
                _f,
                c.flatten(),
                jac=_g,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                options={'maxiter': max_iter_per_chunk, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                c = c_new
            else:
                # If no improvement, accept anyway to move out of local minima?
                # Or keep best? Usually keep best for stability.
                # Let's move to c_new to explore, but track best.
                c = c_new
                
        except Exception as e:
            # If optimization fails, break or continue
            break
        
        # Increase beta to sharpen the soft-min
        beta *= 1.4
        if beta > 300:
            beta = 300
            
        # Re-triangulate periodically
        step_count += 1
        if step_count % retri_freq == 0:
            try:
                # Add tiny noise to avoid degenerate triangulations
                c_perturbed = c + np.random.normal(0, 1e-8, c.shape)
                c_perturbed = np.clip(c_perturbed, -0.5 + 1e-12, 0.5 - 1e-12)
                tri = Delaunay(c_perturbed)
                edges = _get_delaunay_edges(tri)
            except Exception:
                pass
                
    return best_c, best_r
>>>>>>> REPLACE

