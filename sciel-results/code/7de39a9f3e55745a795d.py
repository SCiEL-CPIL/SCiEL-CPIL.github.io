<<<<<<< SEARCH
def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we look for lower (more negative) values
    
    # Perform multiple beam search runs with different beam widths and seeds
    # Then refine with local search
    max_iterations = 100
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Use different beam widths for diversity
        beam_width = 50 + rng.randint(0, 50)
        
        # 1. Beam search
        chain = _beam_search(sequence, dimension, beam_width=beam_width, rng=rng)
        
        if chain is None or len(chain) != n:
            continue
        
        # 2. Local Search
        # Adjust local search steps based on remaining time and sequence length
        chain, energy = _local_search(sequence, chain, rng, max_steps=300)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
=======
def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we look for lower (more negative) values
    
    # Perform multiple beam search runs with different beam widths and seeds
    # Then refine with local search
    max_iterations = 200
    iterations = 0
    
    # Adaptive beam widths: start wider, narrow down
    beam_widths = [100, 80, 60, 50, 40, 30, 25, 20]
    
    while _time_remaining() > 3.0 and iterations < max_iterations:
        iterations += 1
        
        # Use different beam widths for diversity
        bw_idx = min(iterations // 10, len(beam_widths) - 1)
        beam_width = beam_widths[bw_idx] + rng.randint(-10, 10)
        beam_width = max(10, beam_width)
        
        # 1. Beam search
        chain = _beam_search(sequence, dimension, beam_width=beam_width, rng=rng)
        
        if chain is None or len(chain) != n:
            continue
        
        # 2. Local Search
        # Adjust local search steps based on remaining time and sequence length
        time_left = _time_remaining()
        if time_left > 50:
            max_steps = 500
        elif time_left > 20:
            max_steps = 300
        else:
            max_steps = 150
        
        chain, energy = _local_search(sequence, chain, rng, max_steps=max_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
>>>>>>> REPLACE
