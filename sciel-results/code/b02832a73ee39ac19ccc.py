# EVOLVE-BLOCK-START

def _max_radius(pos, i, n):
    """
    Compute the maximum radius for circle i, given the positions of all n circles.
    pos: (n, 2) array of positions.
    i: index of the circle to compute radius for.
    n: total number of circles.
    Returns: max possible radius for circle i.
    """
    x, y = pos[i]
    
    # Constraint from walls
    r_wall = min(x, 1.0 - x, y, 1.0 - y)
    if r_wall <= 0:
        return 0.0
        
    r_max = r_wall
    
    # Constraint from other circles
    for j in range(n):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # The distance between centers must be >= r_i + r_j.
        # We don't know r_j yet if we are computing sequentially, but in this "max radius" 
        # heuristic for local search, we typically assume other circles have their current 
        # "max radius" or we iterate. 
        # However, the standard efficient approach for this specific heuristic is:
        # r_i = min(r_wall, min_j (dist_ij - r_j_current))
        # But we don't have r_j_current stored easily if we recalc.
        # Let's stick to a simpler valid lower bound or a specific iterative scheme.
        # Actually, the inspiration suggests _max_radius is used in a loop.
        # A common trick: r_i = min(r_wall, min_j (dist_ij - r_j_est))
        # If we don't have r_j_est, we can't do this directly.
        
        # Alternative interpretation from inspirations:
        # The local search moves positions. The score is sum of radii.
        # To evaluate a move, we need to know if the configuration is valid for some radii.
        # Usually, we maintain radii. 
        # Let's look at the "Two-phase pipeline" finding: 
        # "fast vectorized force relaxation ... then constraint-aware local search with radius maximization"
        
        # Let's implement a version where we maintain radii and try to increase them.
        # But the function signature in inspiration 1/2 implies _max_radius(pos, i) or similar.
        # If we just have positions, what is the "max radius"?
        # It's the radius such that the circle is disjoint from walls and *other circles at their current radii*.
        # So we need current radii.
        
        # Let's change the strategy slightly to be robust:
        # We will maintain `radii` array.
        # _max_radius will take pos, i, radii.
        pass

    return r_max

def _max_radius_with_others(pos, i, n, radii):
    """
    Compute the maximum radius for circle i, given positions and current radii of all circles.
    """
    x, y = pos[i]
    r_wall = min(x, 1.0 - x, y, 1.0 - y)
    if r_wall <= 0:
        return 0.0
        
    r_max = r_wall
    
    for j in range(n):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # dist >= r_i + r_j  =>  r_i <= dist - r_j
        r_constrained = dist - radii[j]
        if r_constrained < r_max:
            r_max = r_constrained
            
    if r_max < 0:
        return 0.0
    return r_max

def _score_configuration(pos, radii, n):
    """
    Calculate the sum of radii, ensuring validity.
    If invalid, returns -inf.
    """
    # Check walls
    for i in range(n):
        r = radii[i]
        if pos[i, 0] - r < -1e-9 or 1.0 - pos[i, 0] - r < -1e-9:
            return -np.inf
        if pos[i, 1] - r < -1e-9 or 1.0 - pos[i, 1] - r < -1e-9:
            return -np.inf
            
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return -np.inf
                
    return np.sum(radii)

def _local_search(pos, radii, n, time_limit):
    """
    Perform local search to improve the sum of radii.
    Strategy: 
    1. Try to increase radii to their max possible given current positions.
    2. Try moving circles to increase the potential for larger radii.
    """
    current_score = _score_configuration(pos, radii, n)
    if current_score == -np.inf:
        # Try to repair
        for i in range(n):
            radii[i] = _max_radius_with_others(pos, i, n, radii)
        current_score = _score_configuration(pos, radii, n)
        if current_score == -np.inf:
            return pos, radii, 0.0

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Optimization loop
    # We will try small perturbations to positions and then maximize radii.
    # Or, we can use a gradient-like approach on the "max radius" function.
    
    # Simpler and effective: Random restarts of local moves.
    rng = np.random.RandomState(123)
    
    # Step size schedule
    base_step = 0.05
    step = base_step
    
    for iteration in range(1000): # Cap iterations
        if time.time() > start_time + time_limit:
            break
            
        # 1. Maximize radii for current positions
        # Iterate a few times to converge radii
        for _ in range(3):
            for i in range(n):
                radii[i] = _max_radius_with_others(pos, i, n, radii)
        
        score = _score_configuration(pos, radii, n)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
            
        if score == -np.inf:
            # If invalid, revert to best
            pos = best_pos.copy()
            radii = best_radii.copy()
            continue

        # 2. Try a random move
        # Pick a random circle and move it
        i = rng.randint(n)
        
        # Determine direction
        # Heuristic: Move away from the most constraining neighbor or wall?
        # Or just random walk.
        angle = rng.uniform(0, 2 * np.pi)
        dx = np.cos(angle) * step
        dy = np.sin(angle) * step
        
        new_pos = pos.copy()
        new_pos[i, 0] += dx
        new_pos[i, 1] += dy
        
        # Clamp to [0,1]
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0, 1)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0, 1)
        
        # Evaluate new position with current radii (or maximized radii)
        # To be fair, we should maximize radii at the new position too.
        # But that's expensive. Let's just check if current radii fit.
        # If they fit, we accept and then maximize.
        # If not, we might reject.
        
        # Quick check: does the circle still fit with its current radius?
        r_i = radii[i]
        if (new_pos[i, 0] - r_i < -1e-9 or 1.0 - new_pos[i, 0] - r_i < -1e-9 or
            new_pos[i, 1] - r_i < -1e-9 or 1.0 - new_pos[i, 1] - r_i < -1e-9):
            continue # Rejected by walls
            
        # Check against neighbors
        valid = True
        for j in range(n):
            if i == j: continue
            dx = new_pos[i, 0] - pos[j, 0]
            dy = new_pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                break
        if not valid:
            continue
            
        # Accept move
        pos = new_pos
        
        # Reduce step size if we accepted? Or keep it?
        # Let's decay step size slowly
        step = max(0.001, step * 0.99)
        
    # Final maximization
    for _ in range(5):
        for i in range(n):
            radii[i] = _max_radius_with_others(pos, i, n, radii)
            
    final_score = _score_configuration(pos, radii, n)
    if final_score > best_score:
        best_score = final_score
        best_pos = pos.copy()
        best_radii = radii.copy()
        
    return best_pos, best_radii, best_score

def _make_initial_grid(n):
    """Generate a grid-like initial configuration."""
    # Arrange in a grid
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        row = i // side
        col = i % side
        # Center in cell
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        pos[i] = [x, y]
        # Initial radius: half the cell size, but constrained by neighbors
        r = 0.5 / side * 0.8 # Slightly smaller to avoid immediate overlap
        radii[i] = r
    return pos, radii

def _make_initial_random(n, seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    # Small radii to start
    radii = np.full(n, 0.05)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    n = N
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Try different initializations
    inits = [
        ("grid", _make_initial_grid(n)),
        ("rand1", _make_initial_random(n, 1)),
        ("rand2", _make_initial_random(n, 2)),
        ("rand3", _make_initial_random(n, 3)),
    ]
    
    # Add more random seeds if time permits
    for k in range(5):
        inits.append((f"rand_{k+10}", _make_initial_random(n, 10+k)))
        
    for name, (pos, radii) in inits:
        if time.time() > deadline - 5.0: # Need at least 5s for local search
            break
            
        # Run local search
        # Allocate remaining time / number of remaining inits?
        # Or fixed time per init?
        # Let's give each init a chunk of time, e.g., 10s
        # But we have ~7 inits. 7*10 = 70s. Safe.
        time_limit = 10.0
        if time.time() + time_limit > deadline:
            time_limit = deadline - time.time()
            if time_limit < 1.0:
                break
                
        pos_ls, radii_ls, score = _local_search(pos, radii, n, time_limit)
        
        if score > best_score:
            best_score = score
            best_pos = pos_ls.copy()
            best_radii = radii_ls.copy()
            
    # Fallback if no valid solution found
    if best_pos is None or best_score <= 0:
        pos, radii = _make_initial_grid(n)
        # Maximize radii once
        for _ in range(10):
            for i in range(n):
                radii[i] = _max_radius_with_others(pos, i, n, radii)
        best_pos = pos
        best_radii = radii
        best_score = _score_configuration(pos, radii, n)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
