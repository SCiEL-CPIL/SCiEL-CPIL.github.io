<<<<<<< SEARCH
def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation with more diversity
        if r % 5 == 0:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 1:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=2.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 2:
            init_chain, init_energy = _greedy_grow_compact(sequence, rng)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 3:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        else:
            # Greedy with higher temperature
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=5.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        
        # Run local search with more aggressive settings for later restarts
        aggressive = r >= n_restarts // 2
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng, aggressive=aggressive)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy
=======
def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation with more diversity
        if r % 5 == 0:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 1:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=2.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 2:
            init_chain, init_energy = _greedy_grow_compact(sequence, rng)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 3:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        else:
            # Greedy with higher temperature
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=5.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        
        # Run local search with more aggressive settings for later restarts
        aggressive = r >= n_restarts // 3
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng, aggressive=aggressive)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy
>>>>>>> REPLACE

