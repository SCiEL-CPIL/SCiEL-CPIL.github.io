<<<<<<< SEARCH
def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Model-and-optimize approach
    best_priority = _fit_and_optimize(times, predecessors, cycle_time, deadline)
    
    # If time remains, try a few more random perturbations of the best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        current_best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
        
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            # Small random perturbation
            perturbation = rng.randn(n) * 0.1
            trial_priority = best_priority + perturbation
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_priority = trial_priority
                if current_best_score == 1:
                    break
    
    return best_priority
# EVOLVE-BLOCK-END
=======
def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Model-and-optimize approach
    best_priority = _fit_and_optimize(times, predecessors, cycle_time, deadline)
    
    # If time remains, try a few more random perturbations of the best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        current_best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
        
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            # Small random perturbation
            perturbation = rng.randn(n) * 0.1
            trial_priority = best_priority + perturbation
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_priority = trial_priority
                if current_best_score == 1:
                    break
    
    return best_priority
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
