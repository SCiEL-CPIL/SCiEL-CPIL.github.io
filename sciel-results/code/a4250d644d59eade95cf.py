<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 20:
            break
        
        pos_opt, radii_opt, score = _refine_local(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Try differential evolution if time permits
    if time.time() < deadline - 25:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
            
            # Refine the DE result locally as well
            if time.time() < deadline - 10:
                pos_opt, radii_opt, score = _refine_local(best_pos, best_radii, deadline)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = radii_opt.copy()

    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final high-precision polish on the very best candidate
    if time.time() < deadline - 2:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            # One more Nelder-Mead pass with very tight tolerances
            result = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'fatol': 1e-14, 'xtol': 1e-14})
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Corner-heavy initialization (from inspiration)
    pos_ch = np.zeros((N, 2))
    radii_ch = np.zeros(N)
    # Place 4 large circles in corners
    corner_r = 0.15
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos_ch[i, 0] = cx
        pos_ch[i, 1] = cy
        radii_ch[i] = corner_r
    # Fill remaining with smaller circles in a grid
    remaining = N - 4
    grid_size = int(np.ceil(np.sqrt(remaining)))
    for i in range(remaining):
        row = i // grid_size
        col = i % grid_size
        pos_ch[4+i, 0] = 0.3 + col * 0.15
        pos_ch[4+i, 1] = 0.3 + row * 0.15
        radii_ch[4+i] = 0.07
    init_configs.append((pos_ch, radii_ch))
    
    # Run optimization from each initial config
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 40:
            break
        
        pos_opt, radii_opt, score = _refine_local(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Try differential evolution if time permits
    if time.time() < deadline - 35:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
            
            # Refine the DE result locally as well
            if time.time() < deadline - 15:
                pos_opt, radii_opt, score = _refine_local(best_pos, best_radii, deadline)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = radii_opt.copy()
    
    # Greedy independent set refinement: generate many candidates and select best subset
    if time.time() < deadline - 20:
        # Generate a pool of candidate circles from various positions
        candidate_pos = []
        candidate_radii = []
        
        # From current best
        candidate_pos.append(best_pos.copy())
        candidate_radii.append(best_radii.copy())
        
        # Generate random candidates
        rng2 = np.random.RandomState(12345)
        for _ in range(5):
            p = rng2.uniform(0.1, 0.9, size=(N, 2))
            r = rng2.uniform(0.05, 0.15, size=N)
            candidate_pos.append(p)
            candidate_radii.append(r)
        
        # Try greedy selection on each candidate pool
        for ci in range(len(candidate_pos)):
            if time.time() > deadline - 10:
                break
            sel = _greedy_independent_set(candidate_pos[ci], candidate_radii[ci])
            if len(sel) >= N:
                sel_pos = candidate_pos[ci][sel[:N]]
                sel_radii = candidate_radii[ci][sel[:N]]
                sel_pos, sel_radii = _post_process(sel_pos, sel_radii)
                if _is_valid(sel_pos, sel_radii):
                    score = np.sum(sel_radii)
                    if score > best_score:
                        best_score = score
                        best_pos = sel_pos.copy()
                        best_radii = sel_radii.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final high-precision polish: trust-constr then Nelder-Mead
    if time.time() < deadline - 5:
        tc_pos, tc_radii, tc_score = _trust_constr_refine(best_pos, best_radii, deadline)
        if tc_pos is not None and tc_score > best_score:
            best_score = tc_score
            best_pos = tc_pos.copy()
            best_radii = tc_radii.copy()
    
    if time.time() < deadline - 2:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 1500, 'fatol': 1e-14, 'xtol': 1e-14})
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result
>>>>>>> REPLACE
