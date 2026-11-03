<<<<<<< SEARCH
def _nelder_mead_optimize(inner_init, outer_c, rot, target_s, rng, budget):
    """
    Use Nelder-Mead to optimize the positions and angles of inner hexagons.
    """
    x0 = []
    for (x, y, a) in inner_init:
        x0.extend([x, y, a])
    
    def obj(params):
        return _objective(params, outer_c, rot, target_s)
    
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
    
    if time.time() > budget:
        return None
    
    best_params = result.x
    best_obj = result.fun
    
    # Extract inner
    inner_opt = []
    for i in range(N):
        inner_opt.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Compute actual tight side
    s_tight = _min_outer_side(inner_opt, outer_c, rot)
    
    return inner_opt, outer_c, s_tight, rot

def _binary_search_feasible(inner, outer_c, rot, rng, budget):
    """
    Binary search on the outer side length to find the minimum feasible size.
    Uses Nelder-Mead to check feasibility at each size.
    """
    # Initial bounds
    s_low = 3.0
    s_high = 8.0
    
    # First, find a valid configuration at s_high
    best_inner = None
    for attempt in range(5):
        if time.time() > budget:
            break
        inner_try = _nelder_mead_optimize(inner, outer_c, rot, s_high, rng, budget)
        if inner_try is not None:
            inner_opt, oc, s, r = inner_try
            if _valid(inner_opt, oc, s, r) and s <= s_high + 0.1:
                best_inner = inner_opt
                break
    
    if best_inner is None:
        return None
    
    # Binary search
    for _ in range(30):
        if time.time() > budget - 5.0:
            break
        s_mid = (s_low + s_high) / 2.0
        
        inner_try = _nelder_mead_optimize(best_inner, outer_c, rot, s_mid, rng, budget)
        if inner_try is not None:
            inner_opt, oc, s, r = inner_try
            if _valid(inner_opt, oc, s, r) and s <= s_mid + 0.05:
                s_high = s_mid
                best_inner = inner_opt
            else:
                s_low = s_mid
        else:
            s_low = s_mid
    
    # Final result
    if best_inner is not None:
        s_tight = _min_outer_side(best_inner, outer_c, rot)
        if _valid(best_inner, outer_c, s_tight, rot):
            return best_inner, outer_c, s_tight, rot
    
    return None

def _optimize_cluster(inner_init, outer_c, rot, rng, budget):
    """
    Main optimization: use Nelder-Mead to optimize the cluster,
    then binary search on the outer side length.
    """
    # First, get a good initial configuration using Nelder-Mead with a loose bound
    s_loose = 6.0
    result = _nelder_mead_optimize(inner_init, outer_c, rot, s_loose, rng, budget)
    
    if result is None:
        return None
    
    inner_opt, oc, s, r = result
    
    # Now binary search for the minimum feasible size
    bs_result = _binary_search_feasible(inner_opt, oc, r, rng, budget)
    
    if bs_result is not None:
        return bs_result
    
    # Fallback: use the result from Nelder-Mead
    s_tight = _min_outer_side(inner_opt, oc, r)
    if _valid(inner_opt, oc, s_tight, r):
        return inner_opt, oc, s_tight, r
    
    return None
=======
def _local_search(inner_init, outer_c, rot, rng, budget):
    """
    Fast local search using hill climbing with random restarts/perturbations.
    Checks deadline at every step.
    """
    # Deep copy
    current = [list(item) for item in inner_init]
    current_oc = list(outer_c)
    current_rot = rot
    
    # Calculate initial score
    s_curr = _min_outer_side(current, current_oc, current_rot)
    if not _valid(current, current_oc, s_curr, current_rot):
        # If initial is invalid, try to fix it by expanding or moving
        # For now, just return invalid or try a few quick fixes
        pass
        
    best = [list(item) for item in current]
    best_oc = list(current_oc)
    best_rot = current_rot
    best_s = s_curr if _valid(current, current_oc, s_curr, current_rot) else float('inf')
    
    if time.time() > budget:
        return None
        
    temp = 1.0
    cooling = 0.99
    
    while time.time() < budget:
        # Perturb
        new_inner = []
        for i in range(N):
            dx = rng.gauss(0, 0.2)
            dy = rng.gauss(0, 0.2)
            da = rng.gauss(0, 10)
            new_inner.append([current[i][0] + dx, current[i][1] + dy, current[i][2] + da])
        
        new_oc = [current_oc[0] + rng.gauss(0, 0.1), current_oc[1] + rng.gauss(0, 0.1)]
        new_rot = current_rot + rng.gauss(0, 5)
        
        new_s = _min_outer_side(new_inner, new_oc, new_rot)
        
        if _valid(new_inner, new_oc, new_s, new_rot):
            if new_s < best_s:
                best = [list(item) for item in new_inner]
                best_oc = list(new_oc)
                best_rot = new_rot
                best_s = new_s
            
            # Accept if better or with probability exp(-delta/T)
            if new_s <= s_curr or rng.random() < math.exp(-(new_s - s_curr) / temp):
                current = new_inner
                current_oc = new_oc
                current_rot = new_rot
                s_curr = new_s
        else:
            # Invalid: only accept if very low temperature (unlikely) or to escape local minima
            # We generally want to stay valid. If invalid, reject.
            pass
            
        temp *= cooling
        if temp < 0.01:
            break
            
    if best_s == float('inf'):
        return None
        
    return best, best_oc, best_s, best_rot
>>>>>>> REPLACE

