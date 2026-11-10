def fold(sequence, dimension):
    L = len(sequence)
    if L == 0:
        return []
        
    total_time_budget = 12.0 # 12 seconds per sequence
    start_time = time.time()
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345 + len(sequence))
    
    # Run restarts
    while time.time() - start_time < total_time_budget:
        remaining_time = total_time_budget - (time.time() - start_time)
        if remaining_time <= 0.1: # Stop if less than 100ms left
            break
            
        init_chain = _grow_chain(sequence, rng)
        if init_chain is None:
            continue
            
        # Allocate a portion of remaining time to SA
        # If we have 10s left, we can do multiple 1s SA runs.
        # Let's just run SA until time runs out for this restart? 
        # No, we want multiple restarts.
        # Let's fix SA time to 1.0s per restart.
        
        sa_chain, sa_energy = _sa_fold(sequence, init_chain, 1.0, rng)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(L)]
        
    return [list(p) for p in best_chain]
