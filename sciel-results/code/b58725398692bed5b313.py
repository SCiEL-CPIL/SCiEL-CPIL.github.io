<<<<<<< SEARCH
def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simple simulated annealing with adjacent swap and insert moves.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    initial_temp = current_ms * 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        else:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm
=======
def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with adjacent swap and insert moves, using incremental evaluation.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    initial_temp = current_ms * 0.15
    cooling_rate = 0.9995
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        # Recompute prefix/suffix for incremental evaluation
        prefix_comps = _compute_prefix_completions(current_perm, processing_times)
        suffix_comps = _compute_suffix_completions(current_perm, processing_times)
        
        move_type = rng.random()
        if move_type < 0.4:
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
            new_ms = _eval_swap_incremental(current_perm, i, processing_times, prefix_comps, suffix_comps)
        else:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
            new_ms = _eval_insert_incremental(current_perm, i, j, processing_times, prefix_comps, suffix_comps)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm
>>>>>>> REPLACE

