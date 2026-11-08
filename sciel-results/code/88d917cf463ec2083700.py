<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    pairs = _make_pairs(N)
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    # Process each configuration
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        remaining = deadline - elapsed - 2
        
        # Phase 1: Relax overlaps
        time_alloc = min(15.0, remaining * 0.4)
        pos, radii = _relax(pos, radii, pairs, max_iters=3000,
                             time_budget=time_alloc, start_time=start_time)
        
        # Phase 2: Grow radii
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        time_alloc = min(10.0, remaining * 0.3)
        if time_alloc > 1:
            pos, radii = _grow_radii(pos, radii, pairs,
                                      time_budget=time_alloc, start_time=start_time)
        
        # Phase 3: Local search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        time_alloc = min(15.0, remaining * 0.5)
        if time_alloc > 1:
            pos, radii = _local_search(pos, radii, pairs,
                                        time_budget=time_alloc, start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Use best as warm start for next
        if ci < len(configs) - 1 and best_pos is not None and best_score > 0:
            # Keep going with fresh configs for diversity
            pass

    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 3:
            best_pos, best_radii = _local_search(best_pos, best_radii, pairs,
                                                   time_budget=min(20.0, remaining),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
=======
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    pairs = _make_pairs(N)
    
    # Multiple initial configurations - prioritize structured ones
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    for seed in [42, 123, 456]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    # Process each configuration with time allocation
    n_configs = len(configs)
    time_per_config = (deadline - start_time - 10) / n_configs  # reserve 10s for final polish
    
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 8:
            break
        
        config_deadline = min(deadline - 8, start_time + (ci + 1) * time_per_config)
        remaining = config_deadline - elapsed
        
        if remaining < 2:
            break
        
        # Phase 1: Relax overlaps (30% of config time)
        time_alloc = remaining * 0.3
        pos, radii = _relax(pos, radii, pairs, max_iters=5000,
                             time_budget=time_alloc, start_time=start_time)
        
        # Phase 2: Grow radii (30% of config time)
        elapsed = time.time() - start_time
        remaining = config_deadline - elapsed
        time_alloc = remaining * 0.3
        if time_alloc > 0.5:
            pos, radii = _grow_radii(pos, radii, pairs,
                                      time_budget=time_alloc, start_time=start_time)
        
        # Phase 3: Local search (40% of config time)
        elapsed = time.time() - start_time
        remaining = config_deadline - elapsed
        time_alloc = remaining * 0.4
        if time_alloc > 0.5:
            pos, radii = _local_search(pos, radii, pairs,
                                        time_budget=time_alloc, start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Early termination if we've found a very good solution
        if best_score > 2.5:
            break

    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            # Do multiple rounds of local search for polishing
            best_pos, best_radii = _local_search(best_pos, best_radii, pairs,
                                                   time_budget=min(25.0, remaining * 0.6),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
            
            # One more round if time permits
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 2
            if remaining > 3:
                best_pos, best_radii = _local_search(best_pos, best_radii, pairs,
                                                       time_budget=min(10.0, remaining),
                                                       start_time=start_time)
                if _check_validity(best_pos, best_radii):
                    best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
>>>>>>> REPLACE
