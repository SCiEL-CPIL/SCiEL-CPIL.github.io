<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos_grid = _initial_grid(N)
    radii_grid = _compute_radii(pos_grid, N)
    configs.append((pos_grid, radii_grid, "grid"))
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    radii_hex = _compute_radii(pos_hex, N)
    configs.append((pos_hex, radii_hex, "hex"))
    
    # Multiple random starts
    for seed in range(5):
        pos_rand = _initial_random(N, seed)
        radii_rand = _compute_radii(pos_rand, N)
        configs.append((pos_rand, radii_rand, f"rand{seed}"))
    
    # Process each configuration
    for pos, radii, name in configs:
        if time.time() > deadline - 15:
            break
        
        try:
            # Run discrete best-fit optimization
            opt_pos, opt_radii = _discrete_best_fit(pos, radii, N, deadline - 10)
            
            # Validate and fix
            valid_pos, valid_radii, score = _fix_and_validate(opt_pos, opt_radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: small circles in a grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
        
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos_grid = _initial_grid(N)
    radii_grid = _compute_radii(pos_grid, N)
    configs.append((pos_grid, radii_grid, "grid"))
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    radii_hex = _compute_radii(pos_hex, N)
    configs.append((pos_hex, radii_hex, "hex"))
    
    # Multiple random starts (more diverse)
    for seed in range(10):
        pos_rand = _initial_random(N, seed)
        radii_rand = _compute_radii(pos_rand, N)
        configs.append((pos_rand, radii_rand, f"rand{seed}"))
    
    # Process each configuration
    for pos, radii, name in configs:
        if time.time() > deadline - 15:
            break
        
        try:
            # Run discrete best-fit optimization
            opt_pos, opt_radii = _discrete_best_fit(pos, radii, N, deadline - 10)
            
            # Grow radii polish step
            opt_radii = _grow_radii(opt_pos, opt_radii, N, deadline - 5)
            
            # Validate and fix
            valid_pos, valid_radii, score = _fix_and_validate(opt_pos, opt_radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
    
    # Final polish on best result if time remains
    if best_result is not None and time.time() < deadline - 5:
        try:
            best_pos = best_result[:, :2].copy()
            best_radii = best_result[:, 2].copy()
            # One more grow_radii pass
            best_radii = _grow_radii(best_pos, best_radii, N, deadline - 2)
            valid_pos, valid_radii, score = _fix_and_validate(best_pos, best_radii, N)
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: small circles in a grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
        
    return best_result
>>>>>>> REPLACE

