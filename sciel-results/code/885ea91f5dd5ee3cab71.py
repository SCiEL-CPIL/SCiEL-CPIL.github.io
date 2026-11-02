<<<<<<< SEARCH
def _smooth_polish(pts, beta_start=50.0, beta_end=500.0, n_iters=30, time_budget=30.0):
    """Smooth optimization with annealed beta."""
    start = time.time()
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    for it in range(n_iters):
        if time.time() - start > time_budget:
            break
        t = it / max(n_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        def obj_func(p):
            f, g = _objective_and_grad(p, beta)
            return f, g
        
        try:
            res = minimize(obj_func, pts.ravel(), jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-7})
            pts_new = res.x.reshape(N, 3)
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            r_new = _compute_radius(pts_new)
            r_old = _compute_radius(pts)
            if r_new >= r_old:
                pts = pts_new
        except Exception:
            pass
    
    return pts
=======
def _epigraph_polish(pts, beta_start=30.0, beta_end=300.0, n_iters=20, time_budget=20.0):
    """Polish using epigraph formulation with annealed beta."""
    start = time.time()
    r_init = _compute_radius(pts)
    t = r_init  # initial radius estimate
    
    for it in range(n_iters):
        if time.time() - start > time_budget:
            break
        t_frac = it / max(n_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t_frac
        
        def obj_func(x):
            p = x[:N*3].reshape(N, 3)
            t_val = x[N*3]
            f = _epigraph_objective(p, t_val, beta)
            g_pts, g_t = _epigraph_grad(p, t_val, beta)
            g = np.concatenate([g_pts.ravel(), g_t])
            return f, g
        
        x0 = np.concatenate([pts.ravel(), [t]])
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3) + [(0.0, CUBE_HALF)]
        
        try:
            res = minimize(obj_func, x0, jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 30, 'gtol': 1e-8})
            x_new = res.x
            pts_new = x_new[:N*3].reshape(N, 3)
            t_new = x_new[N*3]
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            r_new = _compute_radius(pts_new)
            r_old = _compute_radius(pts)
            if r_new >= r_old - 1e-12:
                pts = pts_new
                t = max(t_new, r_new)
        except Exception:
            pass
    
    return pts
>>>>>>> REPLACE

