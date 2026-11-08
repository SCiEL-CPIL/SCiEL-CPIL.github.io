<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    strategies = ['structured', 'grid', 'random']
    
    # We will allocate time to multiple restarts
    # Let's say we try 5 restarts per strategy initially, then more if time permits
    total_restarts = 15
    
    for restart in range(total_restarts):
        if time.time() > deadline - 10:
            break
        
        strategy = strategies[restart % len(strategies)]
        points = _generate_initial_config(rng, strategy)
        
        # Allocate time for local optimization
        # Use a fixed time slice per restart, e.g., 5 seconds
        opt_points = _local_optimize_heuristic(points, time_limit=5.0)
        
        score = _min_triangle_area_normalized(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
        
        # If we have lots of time left, do more fine tuning on the best found so far
        # But the heuristic optimizer already runs for 5s, so this might be redundant
        # Let's just do more restarts
    
    # Final fine-tuning on the best solution with a longer time budget
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            points_fine = _local_optimize_heuristic(best_points, time_limit=remaining_time)
            score_fine = _min_triangle_area_normalized(points_fine)
            if score_fine > best_score:
                best_score = score_fine
                best_points = points_fine.copy()
    
    if best_points is not None:
        best_points = np.array([_clamp_to_triangle(p) for p in best_points])
        
    return best_points
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    strategies = ['structured', 'lattice', 'random']
    
    # Multi-start with Nelder-Mead optimization
    # Each restart: generate initial config, run NM optimizer
    total_restarts = 20
    
    for restart in range(total_restarts):
        if time.time() > deadline - 15:
            break
        
        strategy = strategies[restart % len(strategies)]
        points = _generate_initial_config(rng, strategy)
        
        # Allocate time for Nelder-Mead optimization
        # Use a fixed time slice per restart
        time_per_restart = 4.0
        opt_points = _local_optimize_nm(points, time_limit=time_per_restart, seed=restart)
        
        score = _min_triangle_area_normalized(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
    
    # Final fine-tuning on the best solution with remaining time
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            points_fine = _local_optimize_nm(best_points, time_limit=remaining_time, seed=999)
            score_fine = _min_triangle_area_normalized(points_fine)
            if score_fine > best_score:
                best_score = score_fine
                best_points = points_fine.copy()
    
    if best_points is not None:
        best_points = np.array([_clamp_to_triangle(p) for p in best_points])
        
    return best_points
>>>>>>> REPLACE

