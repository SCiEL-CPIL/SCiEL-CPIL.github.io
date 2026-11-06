<<<<<<< SEARCH
def _local_optimize_sa(sequence, chain, max_iters=300, rng_seed=42):
    """Local optimization with simulated annealing to escape local minima."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    rng = random.Random(rng_seed)
    dimension = len(chain[0])
    
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.95
    
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        
        # Choose move type: 30% kink, 40% pivot, 30% reversal
        move_type = rng.random()
        
        new_chain = None
        if move_type < 0.3:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(chain, i, rng, dimension)
        elif move_type < 0.7:
            # Pivot move
            i = rng.randint(2, n - 2)
            new_chain = _pivot_move_fast(chain, i, rng, dimension)
        else:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, min(i + 5, n - 1))
            new_chain = _reversal_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e/T) if worse
        if delta_e < 0 or (T > T_min and rng.random() < pow(2.718281828, -delta_e / T)):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy
=======
def _local_optimize_sa(sequence, chain, max_iters=300, rng_seed=42):
    """Local optimization with simulated annealing to escape local minima."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    rng = random.Random(rng_seed)
    dimension = len(chain[0])
    
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.95
    
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        
        # Choose move type: 30% kink, 40% pivot, 30% reversal
        move_type = rng.random()
        
        new_chain = None
        if move_type < 0.3:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(chain, i, rng, dimension)
        elif move_type < 0.7:
            # Pivot move
            i = rng.randint(2, n - 2)
            new_chain = _pivot_move_fast(chain, i, rng, dimension)
        else:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, min(i + 5, n - 1))
            new_chain = _reversal_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e/T) if worse
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy
>>>>>>> REPLACE

