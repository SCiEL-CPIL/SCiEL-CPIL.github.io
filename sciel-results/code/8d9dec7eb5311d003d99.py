<<<<<<< SEARCH
def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    for _ in range(200):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
=======
def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths."""
    start_time = time.time()
    deadline = start_time + 230.0  # Safety margin for 240s budget
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0  # 0 is worse than any negative energy
    
    restarts = 0
    # Increase restarts to ensure robustness, limited by time
    while time.time() < deadline:
        chain = _grow(sequence, dimension, rng)
        restarts += 1
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
            
        # If we find a very good solution, we might stop early? 
        # No, keep searching until time is up to be sure.
        
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
