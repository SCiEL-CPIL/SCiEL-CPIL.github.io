<<<<<<< SEARCH
def _simulated_annealing(pt, perm, deadline, max_iter=20000):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random job to move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        temp_perm = current_perm[:i] + current_perm[i+1:]
        
        # Pick random insertion position
        j = random.randint(0, len(temp_perm))
        if j == i:
            continue  # No change
        
        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms
=======
def _simulated_annealing(pt, perm, deadline, max_iter=50000):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.15, 1.0)
    T_min = max(current_ms * 0.005, 0.5)
    cooling = 0.998
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random job to move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        temp_perm = current_perm[:i] + current_perm[i+1:]
        
        # Pick random insertion position
        j = random.randint(0, len(temp_perm))
        if j == i:
            continue  # No change
        
        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
        new_ms = _makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms
>>>>>>> REPLACE

