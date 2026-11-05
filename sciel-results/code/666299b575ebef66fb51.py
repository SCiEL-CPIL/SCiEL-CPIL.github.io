<<<<<<< SEARCH
def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot and crankshaft moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    # Precompute H indices for faster energy check if possible, but _energy is already fast.
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Choose between pivot and crankshaft
        use_crankshaft = rng.random() < 0.3
        
        if use_crankshaft and n > 6:
            # Crankshaft: pick k1, k2 with k2-k1 >= 2
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        else:
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy
=======
def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot and crankshaft moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters - tune based on sequence length
    T_start = 3.0 if n <= 30 else 2.0
    T_end = 0.005
    start_time = time.time()
    
    # Estimate iterations based on time limit and sequence length
    # Shorter sequences: more iterations possible
    est_iters_per_sec = 50000 if n <= 30 else (20000 if n <= 60 else 10000)
    max_iters = int(time_limit * est_iters_per_sec)
    max_iters = max(max_iters, 10000)
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    # Precompute H indices
    h_indices = set(i for i, s in enumerate(sequence) if s == "H")
    
    # Track acceptance rate for adaptive temperature
    accept_count = 0
    total_count = 0
    last_adapt = start_time
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive temperature: adjust if acceptance rate is too high or low
        if iteration > 100 and time.time() - last_adapt > 0.5:
            rate = accept_count / max(total_count, 1)
            if rate > 0.6:
                T *= 0.7  # Cool faster
            elif rate < 0.1:
                T *= 1.3  # Heat up to escape local minima
            accept_count = 0
            total_count = 0
            last_adapt = time.time()
        
        # Choose between pivot and crankshaft
        use_crankshaft = rng.random() < 0.35
        
        if use_crankshaft and n > 6:
            # Crankshaft: pick k1, k2 with k2-k1 >= 2
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        else:
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        total_count += 1
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            accept_count += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy
>>>>>>> REPLACE

