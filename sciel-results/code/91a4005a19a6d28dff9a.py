<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # Strategies to try
    strategies = ['structured', 'random']
    
    # Allocate time for each strategy
    # We will run multiple SA runs
    total_runs = 10
    time_per_run = (deadline - start_time) / total_runs - 5.0 # 5s buffer
    
    for i in range(total_runs):
        if time.time() > deadline - 10.0:
            break
            
        # Alternate strategies
        strategy = strategies[i % len(strategies)]
        
        # Generate initial config
        points_init = _generate_initial_config(rng, strategy)
        
        # Run SA
        run_deadline = time.time() + time_per_run
        points_sa, area_sa = _simulated_annealing(points_init, run_deadline, rng)
        
        score_sa = area_sa / AREA
        
        if score_sa > best_score_global:
            best_score_global = score_sa
            best_points_global = points_sa.copy()
            
    # Final polishing step
    if best_points_global is not None and time.time() < deadline - 5.0:
        # Use Nelder-Mead on the best result found by SA
        # This should be fast if we start near the optimum
        from scipy.optimize import minimize
        
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _clamp_points(pts)
            return -_min_triangle_area_fast(pts)
        
        x0 = best_points_global.flatten()
        
        # Short run of NM
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        points_nm = result.x.reshape(-1, 2)
        points_nm = _clamp_points(points_nm)
        score_nm = _min_triangle_area_normalized(points_nm)
        
        if score_nm > best_score_global:
            best_score_global = score_nm
            best_points_global = points_nm.copy()

    # Ensure final points are valid
    if best_points_global is not None:
        best_points_global = _clamp_points(best_points_global)
        
    return best_points_global
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # Strategies to try
    strategies = ['structured', 'random']
    
    # Allocate time for multiple force-directed runs
    total_runs = 8
    time_per_run = (deadline - start_time - 10.0) / total_runs  # 10s buffer for polishing
    
    for i in range(total_runs):
        if time.time() > deadline - 15.0:
            break
            
        # Alternate strategies
        strategy = strategies[i % len(strategies)]
        
        # Generate initial config
        points_init = _generate_initial_config(rng, strategy)
        
        # Run force-directed search
        run_deadline = time.time() + time_per_run
        points_fd, area_fd = _force_directed_search(points_init, run_deadline, rng)
        
        score_fd = area_fd / AREA
        
        if score_fd > best_score_global:
            best_score_global = score_fd
            best_points_global = points_fd.copy()
            
    # Final polishing step with Nelder-Mead
    if best_points_global is not None and time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _clamp_points(pts)
            return -_min_triangle_area_fast(pts)
        
        x0 = best_points_global.flatten()
        
        # Run NM for polishing
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        points_nm = result.x.reshape(-1, 2)
        points_nm = _clamp_points(points_nm)
        score_nm = _min_triangle_area_normalized(points_nm)
        
        if score_nm > best_score_global:
            best_score_global = score_nm
            best_points_global = points_nm.copy()

    # Ensure final points are valid
    if best_points_global is not None:
        best_points_global = _clamp_points(best_points_global)
        
    return best_points_global
>>>>>>> REPLACE
