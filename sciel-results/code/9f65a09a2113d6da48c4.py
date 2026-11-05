"""HP lattice protein folding using Multi-Restart Simulated Annealing."""
import random
import time
import math

def _compute_energy_opt(sequence, chain, h_set, dimension):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if i in h_set:
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i and j in h_set:
                        contacts += 1
    return -contacts

def _generate_random_saw(length, dimension):
    """Generate a random self-avoiding walk of given length."""
    if length <= 0:
        return []
    chain = [tuple([0]*dimension)]
    visited = {chain[0]}
    
    for _ in range(length - 1):
        last = chain[-1]
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(last)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in visited:
                    neighbors.append(nb)
        if not neighbors:
            return None # Failed to generate
        next_pos = random.choice(neighbors)
        chain.append(next_pos)
        visited.add(next_pos)
        
    return chain

def _try_end_pivot(chain, k, dimension):
    """
    Try an end-pivot move at index k.
    Keeps chain[0..k] fixed.
    Regenerates chain[k+1..n-1] as a random SAW starting from chain[k].
    Returns new_chain or None if invalid.
    """
    n = len(chain)
    if k >= n - 1:
        return None
        
    fixed_head = chain[:k+1]
    head_set = set(fixed_head)
    
    current_pos = fixed_head[-1]
    new_tail = [current_pos]
    visited_tail = {current_pos}
    
    for _ in range(n - k - 1):
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(current_pos)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in head_set and nb not in visited_tail:
                    neighbors.append(nb)
                    
        if not neighbors:
            return None # Stuck
            
        current_pos = random.choice(neighbors)
        new_tail.append(current_pos)
        visited_tail.add(current_pos)
        
    return fixed_head + new_tail[1:]

def _local_search(sequence, chain, dimension, h_set, time_limit):
    """
    Perform local search to improve the chain.
    """
    start_time = time.time()
    n = len(chain)
    current_energy = _compute_energy_opt(sequence, chain, h_set, dimension)
    best_chain = chain[:]
    best_energy = current_energy
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try pivots from end to start
        for k in range(n - 2, 0, -1):
            if time.time() - start_time >= time_limit:
                break
            # Try a few random regenerations
            for _ in range(5):
                new_chain = _try_end_pivot(chain, k, dimension)
                if new_chain is None:
                    continue
                new_energy = _compute_energy_opt(sequence, new_chain, h_set, dimension)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                    break # Move to next k
    return best_chain, best_energy

def _simulated_annealing(sequence, dimension, time_limit, rng_seed=None, h_set=None):
    """Run a single simulated annealing session."""
    if rng_seed is not None:
        random.seed(rng_seed)
        
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)], 0

    # Initial state
    chain = None
    for _ in range(20):
        chain = _generate_random_saw(n, dimension)
        if chain:
            break
    if not chain:
        chain = [(i, 0, 0) for i in range(n)]
        
    current_energy = _compute_energy_opt(sequence, chain, h_set, dimension)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Cooling parameters
    T_start = 3.0
    T_end = 0.01
    
    start_time = time.time()
    steps = 0
    # Estimate steps per second to set total steps
    # Assume 100k steps/sec is achievable with optimized code
    target_steps = int(time_limit * 50000) # Conservative estimate
    
    while steps < target_steps and time.time() - start_time < time_limit:
        # Pick random pivot point
        k = random.randint(1, n - 2)
        
        new_chain = _try_end_pivot(chain, k, dimension)
        
        if new_chain is None:
            steps += 1
            continue
            
        new_energy = _compute_energy_opt(sequence, new_chain, h_set, dimension)
        delta_e = new_energy - current_energy
        
        # Calculate current temperature
        # Geometric cooling
        if steps == 0:
            T = T_start
        else:
            # Cool over target_steps
            cool_rate = (T_end / T_start) ** (1.0 / max(1, target_steps))
            T = T_start * (cool_rate ** steps)
            
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        steps += 1
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    # Precompute H set
    h_set = {i for i, res in enumerate(sequence) if res == 'H'}
    
    # Time budget allocation
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    
    best_chain_global = None
    best_energy_global = 0
    
    # Run multiple episodes
    while time.time() - start_time < time_budget * 0.8:
        remaining = time_budget * 0.8 - (time.time() - start_time)
        if remaining <= 0:
            break
            
        episode_time = min(1.0, remaining)
        
        chain, energy = _simulated_annealing(sequence, dimension, episode_time, h_set=h_set)
        
        if best_energy_global == 0 or energy < best_energy_global:
            best_energy_global = energy
            best_chain_global = chain
            
    if best_chain_global is None:
        chain = _generate_random_saw(n, dimension)
        if not chain:
            chain = [(i, 0, 0) for i in range(n)]
        best_chain_global = chain
        
    # Final Local Search Refinement
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 0.5:
        best_chain_global, best_energy_global = _local_search(
            sequence, best_chain_global, dimension, h_set, remaining_time
        )
        
    return [list(p) for p in best_chain_global]
