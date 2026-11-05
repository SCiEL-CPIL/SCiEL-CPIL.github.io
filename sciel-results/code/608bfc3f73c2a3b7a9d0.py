"""HP lattice protein folding using Multi-Restart Simulated Annealing."""
import random
import time
import math

def _compute_energy(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
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

def _simulated_annealing(sequence, dimension, time_limit, rng_seed=None):
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
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Cooling parameters
    # Max possible energy change is roughly n/2, but typical moves change by 1-3.
    # Start T high enough to accept bad moves, end T low.
    T_start = 3.0
    T_end = 0.01
    
    start_time = time.time()
    steps = 0
    stagnation = 0
    max_stagnation = 10000 # Restart logic inside SA if stuck? Or just let it cool?
    
    while time.time() - start_time < time_limit:
        # Adaptive cooling: reduce T as we go
        # Simple linear or geometric. Geometric is better.
        # Let's estimate total steps based on time? Hard.
        # Just use a fixed decay rate per step.
        # If we run for ~1s, we might do 100k-1M steps.
        # Let's use a decay rate that cools down over ~50k steps.
        # T = T_start * (T_end/T_start)^(steps / 50000)
        
        # Pick random pivot point
        k = random.randint(1, n - 2)
        
        new_chain = _try_end_pivot(chain, k, dimension)
        
        if new_chain is None:
            # Failed to generate a valid tail
            # Skip
            steps += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Calculate current temperature
        # We need a reference for steps. Let's just use a simple exponential decay based on time?
        # No, step-based is better for SA.
        # Let's assume we can do ~50000 steps in 1 second.
        # So after 1 second, T should be close to T_end.
        # Let's use a dynamic decay.
        
        # Simpler: Just use a fixed decay rate per step, e.g., 0.9995
        # This might not cool fast enough for long runs.
        
        # Let's use time-based cooling?
        # T = T_start * exp(-time_elapsed * cooling_const)
        
        elapsed = time.time() - start_time
        # We want T to drop from 3.0 to 0.01 in ~0.5 * time_limit
        # ln(3.0/0.01) = -cooling_const * 0.5 * time_limit
        # cooling_const = ln(300) / (0.5 * time_limit)
        
        if time_limit > 0:
            cooling_const = math.log(T_start / T_end) / (0.5 * time_limit)
            T = T_start * math.exp(-cooling_const * elapsed)
        else:
            T = T_end
            
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
            
        steps += 1
        
        # Optional: Restart within SA if stuck too long?
        # If stagnation > 5000 and T is still high, we might restart the chain.
        # But restarting the chain is expensive.
        # Let's just rely on the outer loop restarts.
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    # Time budget allocation
    # Total budget ~240s for 26 sequences.
    # 10 seqs of ~48, 6 seqs of ~64-90, 10 seqs of ~100-136.
    # Allocate based on length.
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    
    best_chain_global = None
    best_energy_global = 0 # Energy is negative, so 0 is "worse" than any negative energy
    # Actually, initialize to a valid chain's energy
    
    # Run multiple episodes
    # Each episode gets a fraction of the time.
    # Let's run episodes of ~1 second each, or until time runs out.
    
    while time.time() - start_time < time_budget * 0.95:
        # Determine time for this episode
        # If we have > 2s left, do a 1s episode.
        # If we have < 2s left, use the rest.
        remaining = time_budget * 0.95 - (time.time() - start_time)
        if remaining <= 0:
            break
            
        episode_time = min(1.0, remaining)
        
        # Run one SA episode
        # Use a random seed for reproducibility in debugging, but random is fine
        chain, energy = _simulated_annealing(sequence, dimension, episode_time)
        
        if best_energy_global == 0 or energy < best_energy_global:
            best_energy_global = energy
            best_chain_global = chain
            
    if best_chain_global is None:
        # Fallback
        chain = _generate_random_saw(n, dimension)
        if not chain:
            chain = [(i, 0, 0) for i in range(n)]
        best_chain_global = chain
        
    return [list(p) for p in best_chain_global]
