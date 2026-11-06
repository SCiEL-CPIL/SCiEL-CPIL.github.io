<<<<<<< SEARCH
def design(problem):
    """
    Design a 32x32 photonic crystal unit cell for a large TM band gap.
    Uses a 2x2 array of rods and local search.
    """
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 180.0
    
    best_params = {
        'radius': 2.0,
        'offset': 6.0
    }
    
    best_cell = _generate_cell(best_params, size)
    best_score = _compute_gap_fast(best_cell, size, eps_high=eps_high)
    
    param_names = ['radius', 'offset']
    step_sizes = {'radius': 0.2, 'offset': 0.5}
    
    bounds = {
        'radius': (1.0, 4.0),
        'offset': (2.0, 10.0)
    }
    
    n_restarts = 5
    max_iter = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params = dict(best_params)
        else:
            params = {}
            for name in param_names:
                lo, hi = bounds[name]
                params[name] = lo + (hi - lo) * (0.2 + 0.6 * (restart / n_restarts) * (0.5 + 0.5 * math.sin(restart * 7.0)))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for name in param_names:
                if time.time() > deadline:
                    break
                
                step = step_sizes[name]
                for direction in [-1, 1]:
                    new_params = dict(params)
                    new_val = params[name] + direction * step
                    lo, hi = bounds[name]
                    if new_val < lo or new_val > hi:
                        continue
                    new_params[name] = new_val
                    
                    new_cell = _generate_cell(new_params, size)
                    new_score = _compute_gap_fast(new_cell, size, eps_high=eps_high)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_params = dict(new_params)
                        best_cell = new_cell
                        params = dict(new_params)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step_scale = 0.5
                for name in param_names:
                    if time.time() > deadline:
                        break
                    step = step_sizes[name] * step_scale
                    for direction in [-1, 1]:
                        new_params = dict(params)
                        new_val = params[name] + direction * step
                        lo, hi = bounds[name]
                        if new_val < lo or new_val > hi:
                            continue
                        new_params[name] = new_val
                        
                        new_cell = _generate_cell(new_params, size)
                        new_score = _compute_gap_fast(new_cell, size, eps_high=eps_high)
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_params = dict(new_params)
                            best_cell = new_cell
                            params = dict(new_params)
                            improved = True
                            break
                    if improved:
                        break
    
    return best_cell
=======
def design(problem):
    """
    Design a 32x32 photonic crystal unit cell for a large TM band gap.
    Uses exhaustive search over 16 possible 4-bit patterns.
    """
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 180.0
    
    best_score = -np.inf
    best_cell = None
    
    # Exhaustively evaluate all 16 patterns
    for pattern in range(16):
        if time.time() > deadline:
            break
        
        params = {'pattern': pattern}
        cell = _generate_cell(params, size)
        score = _compute_gap_fast(cell, size, eps_high=eps_high)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If no cell was found (shouldn't happen), return a default
    if best_cell is None:
        best_cell = _generate_cell({'pattern': 0}, size)
    
    return best_cell
>>>>>>> REPLACE
