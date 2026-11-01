def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    fib_pts = _fib_sample(N)
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    for seed_offset in range(3):
        np.random.seed(42 + seed_offset)
        rand_pts = np.random.randn(N, 3)
        rand_pts = np.array([_normalize(p) for p in rand_pts])
        initial_configs.append(rand_pts)
    
    best_config = initial_configs[0].copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Repulsion Relaxation to spread points
    for config in initial_configs:
        if time.time() > deadline - 40.0:
            break
        relaxed = _repulsion_relax(config, alpha_init=0.05, decay=0.995, max_iter=300, deadline=deadline - 40.0)
        score = _covering_radius_deg_hull(relaxed)
        if score < best_score:
            best_score = score
            best_config = relaxed.copy()
    
    # Phase 2: L-BFGS optimization with soft-max objective
    starting_points = [best_config.copy(), _fib_sample(N)]
    
    for start_pts in starting_points:
        if time.time() > deadline - 20.0:
            break
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                res = minimize(_soft_max_objective, start_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200},
                               args=(beta,))
                
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                current_radius = _covering_radius_deg_hull(new_pts)
                if current_radius < best_score:
                    best_score = current_radius
                    best_config = new_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Lloyd's algorithm for local optimization
    if time.time() < deadline - 15.0:
        lloyd_deadline = deadline - 15.0
        config = best_config.copy()
        best_lloyd = config.copy()
        best_lloyd_score = _covering_radius_deg_hull(config)
        
        it = 0
        while time.time() < lloyd_deadline:
            new_config = _lloyd_step_vectorized(config)
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_lloyd_score - 1e-10:
                best_lloyd_score = new_score
                best_lloyd = new_config.copy()
                config = new_config
            else:
                # Perturb to escape local minimum
                idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                rand_dir = _normalize(rand_dir)
                old_pt = config[idx].copy()
                config[idx] = _normalize(config[idx] + 0.05 * rand_dir)
                pert_score = _covering_radius_deg_hull(config)
                if pert_score < best_lloyd_score:
                    best_lloyd_score = pert_score
                    best_lloyd = config.copy()
                else:
                    config[idx] = old_pt
            
            it += 1
            if it > 500:
                break
        
        if best_lloyd_score < best_score:
            best_score = best_lloyd_score
            best_config = best_lloyd.copy()
    
    # Phase 4: Final projected gradient polish
    if time.time() < deadline - 5.0:
        polished = _projected_gradient_polish(best_config, deadline, lr_init=0.005, lr_decay=0.95, max_iter=100)
        final_score = _covering_radius_deg_hull(polished)
        if final_score < best_score:
            best_score = final_score
            best_config = polished.copy()
    
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                best_config[j] = _normalize(best_config[j])
                
    return best_config
