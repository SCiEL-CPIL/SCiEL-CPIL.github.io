<<<<<<< SEARCH
def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Randomized growth restarts
    # Spend about 70% of available time on growth
    growth_deadline = _time_remaining() * 0.7
    
    iterations = 0
    max_iterations = 10000
    
    while _time_remaining() > growth_deadline and iterations < max_iterations:
        iterations += 1
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement on the best solution
    if _time_remaining() > 3.0:
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=300)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Second local search pass with fresh randomness
    if _time_remaining() > 5.0:
        rng2 = random.Random(99999 + len(sequence))
        best_chain = _segment_move(sequence, best_chain, rng2, max_attempts=300)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
=======
def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Randomized growth restarts with periodic local search
    # Spend about 50% of available time on growth
    growth_deadline = _time_remaining() * 0.5
    
    iterations = 0
    max_iterations = 5000
    
    while _time_remaining() > growth_deadline and iterations < max_iterations:
        iterations += 1
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Apply local search every 5 restarts to escape local minima
        if iterations % 5 == 0 and best_chain is not None and _time_remaining() > growth_deadline + 2.0:
            best_chain = _segment_move(sequence, best_chain, rng, max_attempts=50)
            best_energy = _energy(sequence, best_chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement on the best solution
    if _time_remaining() > 5.0:
        best_chain = _segment_move(sequence, best_chain, rng, max_attempts=200)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Second local search pass with fresh randomness
    if _time_remaining() > 10.0:
        rng2 = random.Random(99999 + len(sequence))
        best_chain = _segment_move(sequence, best_chain, rng2, max_attempts=200)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 4: Final refinement with more attempts
    if _time_remaining() > 15.0:
        rng3 = random.Random(77777 + len(sequence))
        best_chain = _segment_move(sequence, best_chain, rng3, max_attempts=300)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
>>>>>>> REPLACE
