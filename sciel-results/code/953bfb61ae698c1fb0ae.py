<<<<<<< SEARCH
    best_c = None
    best_r = 0.0
    
    # Try multiple grid configurations and optimizations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if time.time() - t0 > _DEADLINE:
            break
        
        for jitter in [0.0, 0.002, 0.005, 0.01]:
            if time.time() - t0 > _DEADLINE:
                break
            
            c0 = _make_grid(nx, ny, nz, jitter=jitter)
            c0 = np.clip(c0, -0.49, 0.49)
            
            r_current = _compute_radius(c0)
            if best_r is None or r_current > best_r:
                best_r = r_current
                best_c = c0.copy()
            
            # Incrementally increase radius
            r_target = r_current * 1.003
            max_r_target = 0.1125  # Slightly above known best 0.11138
            
            while time.time() - t0 < _DEADLINE and r_target <= max_r_target:
                c_test, converged = _relax_spheres(c0, r_target, max_iter=2000)
                
                r_actual = _compute_radius(c_test)
                
                if r_actual >= r_target - 1e-9 and r_actual > best_r:
                    best_r = r_actual
                    best_c = c_test.copy()
                    c0 = c_test.copy()
                    r_target *= 1.003
                elif r_actual > best_r:
                    # Even if not fully converged, if we improved, keep it
                    best_r = r_actual
                    best_c = c_test.copy()
                    c0 = c_test.copy()
                    r_target = r_actual * 1.002
                else:
                    # Failed to improve, try smaller increment
                    r_target = (r_target + best_r) / 2.0
                    if r_target <= best_r + 1e-10:
                        break
            
            # After grid optimization, try random perturbations
            if best_c is not None and time.time() - t0 < _DEADLINE:
                for _ in range(3):
                    if time.time() - t0 > _DEADLINE:
                        break
                    c_pert = best_c + rng.randn(*best_c.shape) * 0.008
                    c_pert = np.clip(c_pert, -(0.5 - best_r), 0.5 - best_r)
                    r_pert = _compute_radius(c_pert)
                    if r_pert > best_r:
                        best_r = r_pert
                        best_c = c_pert.copy()
=======
    best_c = None
    best_r = 0.0
    
    # Define a helper to run the optimization loop for a given initial configuration
    def _optimize_config(c_init, r_init):
        nonlocal best_c, best_r
        if r_init > best_r:
            best_r = r_init
            best_c = c_init.copy()
        
        r_target = r_init * 1.003
        max_r_target = 0.1125
        
        while time.time() - t0 < _DEADLINE and r_target <= max_r_target:
            c_test, converged = _relax_spheres(c_init, r_target, max_iter=2000)
            r_actual = _compute_radius(c_test)
            
            if r_actual >= r_target - 1e-9 and r_actual > best_r:
                best_r = r_actual
                best_c = c_test.copy()
                c_init = c_test.copy()
                r_target *= 1.003
            elif r_actual > best_r:
                best_r = r_actual
                best_c = c_test.copy()
                c_init = c_test.copy()
                r_target = r_actual * 1.002
            else:
                r_target = (r_target + best_r) / 2.0
                if r_target <= best_r + 1e-10:
                    break
        
        # Perturbations
        if best_c is not None and time.time() - t0 < _DEADLINE:
            for _ in range(3):
                if time.time() - t0 > _DEADLINE:
                    break
                c_pert = best_c + rng.randn(*best_c.shape) * 0.008
                c_pert = np.clip(c_pert, -(0.5 - best_r), 0.5 - best_r)
                r_pert = _compute_radius(c_pert)
                if r_pert > best_r:
                    best_r = r_pert
                    best_c = c_pert.copy()

    # 1. Try FCC lattice initialization
    if time.time() - t0 < _DEADLINE:
        c_fcc = _make_fcc_grid(3, 100)
        c_fcc = np.clip(c_fcc, -0.49, 0.49)
        r_fcc = _compute_radius(c_fcc)
        _optimize_config(c_fcc, r_fcc)

    # 2. Try multiple grid configurations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if time.time() - t0 > _DEADLINE:
            break
        
        for jitter in [0.0, 0.002, 0.005, 0.01]:
            if time.time() - t0 > _DEADLINE:
                break
            
            c0 = _make_grid(nx, ny, nz, jitter=jitter)
            c0 = np.clip(c0, -0.49, 0.49)
            
            r_current = _compute_radius(c0)
            _optimize_config(c0, r_current)
>>>>>>> REPLACE

I also need to add the `_make_fcc_grid` function.

