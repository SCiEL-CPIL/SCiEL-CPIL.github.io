<<<<<<< SEARCH
def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with insertion perturbations (more effective than swap for flow shop).
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.999
    min_T = 1e-6
    
    while time.time() < deadline and T > min_T:
        # Use insertion move for perturbation
        i, j = random.sample(range(n_jobs), 2)
        if i == j:
            continue
        new_seq = _insertion_move(current_seq, i, j)
        new_makespan = _calc_makespan(processing_times, new_seq)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
    
    return best_seq, best_makespan
=======
def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with adaptive temperature and insertion perturbations.
    Uses a faster cooling schedule and accepts worse solutions more aggressively early on.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Adaptive initial temperature
    T = current_makespan * 0.15
    min_T = 1e-8
    
    # Estimate cooling rate to reach min_T near deadline
    # T * rate^steps = min_T => rate = (min_T/T)^(1/steps)
    # Assume ~10000 steps
    estimated_steps = 10000
    cooling_rate = (min_T / T) ** (1.0 / estimated_steps) if T > min_T else 0.99
    
    no_improve_streak = 0
    max_no_improve = 500  # Restart if stuck
    
    while time.time() < deadline:
        # Use insertion move for perturbation
        i, j = random.sample(range(n_jobs), 2)
        if i == j:
            continue
        new_seq = _insertion_move(current_seq, i, j)
        new_makespan = _calc_makespan(processing_times, new_seq)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
                no_improve_streak = 0
            else:
                no_improve_streak += 1
        
        T *= cooling_rate
        
        # If stuck, do a random restart from best
        if no_improve_streak > max_no_improve:
            current_seq = list(best_seq)
            current_makespan = best_makespan
            T = best_makespan * 0.1  # Reset temperature
            no_improve_streak = 0
    
    return best_seq, best_makespan
>>>>>>> REPLACE

