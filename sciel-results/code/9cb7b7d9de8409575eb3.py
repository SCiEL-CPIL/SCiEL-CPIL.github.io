def _local_search(c, deadline, iters=4000):
    """Local search: Nelder-Mead on exact radius, then fine-tune."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Nelder-Mead phase
    if time.time() < deadline:
        try:
            from scipy.optimize import minimize
            def exact_obj(x):
                c_x = x.reshape(N, 3)
                return -_compute_radius(c_x)
            
            res = minimize(
                exact_obj,
                c.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10},
            )
            c_nm = np.clip(res.x.reshape(N, 3), -0.499, 0.499)
            r_nm = _compute_radius(c_nm)
            if r_nm > best_r:
                best_r = r_nm
                best_c = c_nm.copy()
        except Exception:
            pass
    
    # Fine-tune phase (existing logic, simplified)
    c = best_c
    rng = np.random.default_rng(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        improved = False
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            
            for step in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = c.copy()
                trial[i] += step * direction
                trial[j] -= step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
        else:
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn
            
            for step in (0.005, 0.002, 0.001, 0.0005, 0.0002):
                trial = c.copy()
                trial[i] += step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
        
        if not improved and it % 100 == 0:
            # Random kick
            trial = c.copy()
            trial += rng.normal(scale=0.01, size=(N, 3))
            trial = np.clip(trial, -0.499, 0.499)
            r_trial = _compute_radius(trial)
            if r_trial > best_r:
                c = trial
                best_r = r_trial
                best_c = c.copy()
    
    return best_c, best_r
