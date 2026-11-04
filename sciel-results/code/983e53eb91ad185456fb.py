<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. More diverse random inits
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]:
        rng_s = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng_s.random()
            v = rng_s.random()
            while u + v > 1.0:
                u = rng_s.random()
                v = rng_s.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Structured inits based on symmetry
    # Regular 11-point pattern
    magic_pts_1 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts_1).flatten())
    
    # Another structured pattern
    magic_pts_2 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.0], [0.66, 0.0], [0.0, 0.33], [0.0, 0.66],
        [0.33, 0.33], [0.66, 0.33], [0.33, 0.66], [0.5, 0.5],
    ]
    inits.append(np.array(magic_pts_2).flatten())
    
    # Grid-based init
    grid_pts = []
    for i in range(4):
        for j in range(4 - i):
            grid_pts.append([i/3.0, j/3.0])
    if len(grid_pts) >= N:
        inits.append(np.array(grid_pts[:N]).flatten())
    
    # Process each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            x_opt, score = _multi_start_optimize(init_bary, deadline - 10)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    if best_x is None:
        best_x = _structured_init_fps()
        best_score = _min_area_ratio_from_bary(best_x)
    
    # Simulated Annealing Phase
    if time.time() < deadline - 25:
        try:
            sa_x, sa_score = _simulated_annealing(best_x, deadline - 15)
            if sa_score > best_score:
                best_score = sa_score
                best_x = sa_x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        try:
            # Polish with epigraph
            polish_x = _local_optimize_epigraph(best_x, deadline - 5)
            score = _min_area_ratio_from_bary(polish_x)
            if score > best_score:
                best_score = score
                best_x = polish_x
            
            # Additional smooth min polish
            bounds = [(0, 1)] * (2 * N)
            current_x = best_x.flatten()
            for beta in [5000.0, 20000.0]:
                if time.time() > deadline - 3:
                    break
                res = minimize(
                    _objective_smooth,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                polish_x = res.x.reshape(N, 2)
                polish_x = _project_to_bary_simplex(polish_x)
                score = _min_area_ratio_from_bary(polish_x)
                if score > best_score:
                    best_score = score
                    best_x = polish_x
                    current_x = polish_x.flatten()
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
=======
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (evaluator kills at 150s)
    
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. More diverse random inits (fewer but better distributed)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        rng_s = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng_s.random()
            v = rng_s.random()
            while u + v > 1.0:
                u = rng_s.random()
                v = rng_s.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Structured inits based on symmetry
    # Regular 11-point pattern
    magic_pts_1 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts_1).flatten())
    
    # Another structured pattern
    magic_pts_2 = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.0], [0.66, 0.0], [0.0, 0.33], [0.0, 0.66],
        [0.33, 0.33], [0.66, 0.33], [0.33, 0.66], [0.5, 0.5],
    ]
    inits.append(np.array(magic_pts_2).flatten())
    
    # Grid-based init
    grid_pts = []
    for i in range(4):
        for j in range(4 - i):
            grid_pts.append([i/3.0, j/3.0])
    if len(grid_pts) >= N:
        inits.append(np.array(grid_pts[:N]).flatten())
    
    # 4. Symmetric hexagonal-like init
    hex_init = [
        [1/3, 1/3],
        [0.15, 0.15], [0.85, 0.15], [0.15, 0.85],
        [0.5, 0.05], [0.05, 0.5], [0.95, 0.05],
        [0.3, 0.4], [0.7, 0.3], [0.4, 0.7], [0.55, 0.55],
    ]
    hex_init_arr = np.array(hex_init)
    hex_init_arr = _project_to_bary_simplex(hex_init_arr)
    inits.append(hex_init_arr.flatten())
    
    # Process each init with time budget
    n_inits = len(inits)
    time_per_init = (deadline - 40) / n_inits  # Reserve 40s for SA and polish
    
    for idx, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        init_deadline = min(deadline - 40, start_time + (idx + 1) * time_per_init + 5)
        
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            x_opt, score = _multi_start_optimize(init_bary, init_deadline)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception:
            pass
    
    if best_x is None:
        best_x = _structured_init_fps()
        best_score = _min_area_ratio_from_bary(best_x)
    
    # Simulated Annealing Phase - give it generous time
    sa_deadline = deadline - 20
    if time.time() < sa_deadline - 10:
        try:
            sa_x, sa_score = _simulated_annealing(best_x, sa_deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_x = sa_x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 8:
        try:
            # Polish with epigraph
            polish_x = _local_optimize_epigraph(best_x, deadline - 5)
            score = _min_area_ratio_from_bary(polish_x)
            if score > best_score:
                best_score = score
                best_x = polish_x
            
            # Additional smooth min polish with very high beta
            bounds = [(0, 1)] * (2 * N)
            current_x = best_x.flatten()
            for beta in [5000.0, 20000.0, 50000.0]:
                if time.time() > deadline - 3:
                    break
                try:
                    res = minimize(
                        _objective_smooth,
                        current_x,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-11}
                    )
                    polish_x = res.x.reshape(N, 2)
                    polish_x = _project_to_bary_simplex(polish_x)
                    score = _min_area_ratio_from_bary(polish_x)
                    if score > best_score:
                        best_score = score
                        best_x = polish_x
                        current_x = polish_x.flatten()
                except Exception:
                    pass
        except Exception:
            pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
>>>>>>> REPLACE

