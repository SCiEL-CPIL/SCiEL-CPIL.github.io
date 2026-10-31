# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Check inside square
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _objective(x):
    """Objective: minimize negative sum of radii + penalty for violations (Vectorized)."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Ensure radii are positive for calculation
    radii_safe = np.maximum(radii, 1e-9)
    
    # Boundary Penalties
    p_left = np.maximum(0, radii_safe - pos[:, 0])
    p_right = np.maximum(0, pos[:, 0] - (1.0 - radii_safe))
    p_bottom = np.maximum(0, radii_safe - pos[:, 1])
    p_top = np.maximum(0, pos[:, 1] - (1.0 - radii_safe))
    
    boundary_penalty = np.sum(p_left**2) + np.sum(p_right**2) + np.sum(p_bottom**2) + np.sum(p_top**2)
    
    # Overlap Penalties
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    r_sum = radii_safe[:, np.newaxis] + radii_safe[np.newaxis, :]
    violations = r_sum - dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_violations = np.maximum(0, violations[mask])
    overlap_penalty = np.sum(valid_violations**2)
    
    return -np.sum(radii_safe) + 1000.0 * (boundary_penalty + overlap_penalty)

def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid: 25 circles
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    # 26th circle: place in a corner
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii

def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    # Row 1: 4 circles (offset)
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    # Row 3: 4 circles (offset)
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    # 26th circle
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii

def _initial_radial():
    """Create a radial packing: 1 center, 6 ring 1, 12 ring 2, 7 extra."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.15
    
    # Ring 1: 6 circles
    r1 = 0.15
    d1 = 2 * r1 * 0.95 # slightly smaller to allow growth
    for i in range(6):
        idx = 1 + i
        angle = i * np.pi / 3
        pos[idx, 0] = 0.5 + d1 * np.cos(angle)
        pos[idx, 1] = 0.5 + d1 * np.sin(angle)
        radii[idx] = r1
        
    # Ring 2: 12 circles
    r2 = 0.12
    d2 = d1 + 2 * r2 * 0.95
    # Offset angles to fit in gaps
    for i in range(12):
        idx = 7 + i
        angle = i * np.pi / 6 + np.pi / 6
        pos[idx, 0] = 0.5 + d2 * np.cos(angle)
        pos[idx, 1] = 0.5 + d2 * np.sin(angle)
        radii[idx] = r2
        
    # Extra 7 circles: place in corners or remaining gaps
    # Corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(4):
        idx = 19 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = 0.1
        
    # Mid-edges
    mids = [(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    for i in range(3): # only 3 more needed
        idx = 23 + i
        pos[idx, 0] = mids[i][0]
        pos[idx, 1] = mids[i][1]
        radii[idx] = 0.08
        
    return pos, radii

def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        # Check and fix boundaries
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            # Right
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Bottom
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Top
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        # Check and fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision local search to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
        
        if result.fun < _objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
        
    return pos, radii

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try multiple optimization methods
    methods = ['L-BFGS-B', 'Powell']
    
    for method in methods:
        if time.time() > deadline - 5:
            break
        
        try:
            if method == 'L-BFGS-B':
                result = minimize(_objective, x0, method=method, bounds=bounds,
                                  options={'maxiter': 2000, 'ftol': 1e-12})
            elif method == 'Powell':
                result = minimize(_objective, x0, method=method, bounds=bounds,
                                  options={'maxiter': 1000, 'ftol': 1e-12})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                # Post-process to ensure validity
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead as well
    if time.time() > deadline - 5:
        pass
    else:
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'fatol': 1e-12})
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
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    best_pos = None
    best_radii = None
    best_score = -np.inf

    # Try different initial configurations
    init_configs = []

    # Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii, 'grid'))

    # Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii, 'hex'))
    
    # Radial
    pos, radii = _initial_radial()
    init_configs.append((pos, radii, 'radial'))

    # Random with different seeds
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii, f'rand_{seed}'))

    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii, 'grid_pert'))

    # Run optimization from each initial config
    for pos, radii, name in init_configs:
        if time.time() > deadline - 15:
            break
        
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()

    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None or best_score < 0:
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx, 0] = (i + 0.5) / side
                    pos[idx, 1] = (j + 0.5) / side
                    radii[idx] = 0.09
        # 26th circle
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii

    # Polish the best result
    if time.time() < deadline - 5:
        best_pos, best_radii = _polish(best_pos, best_radii, deadline)
        # Re-verify after polish
        if _is_valid(best_pos, best_radii):
            best_score = np.sum(best_radii)
        else:
            # If polish made it invalid, revert to pre-polish state? 
            # _polish returns original if invalid. So this check is redundant if _polish is correct.
            pass

    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
