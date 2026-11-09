"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_NEIGHBOR_OFFSETS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

def _neighbours(point):
    x, y, z = point
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        yield (x+dx, y+dy, z+dz)

def _energy_with_occupied(sequence, chain, occupied, h_set):
    """Calculate total energy using precomputed occupied map and h_set."""
    contacts = 0
    for i in h_set:
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (x+dx, y+dy, z+dz)
            j = occupied.get(q)
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts

def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    return _energy_with_occupied(sequence, chain, where, h_set)

def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts.
    bias controls the strength of the preference for creating contacts.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0
            if i in h_set:
                x, y, z = site
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (x+dx, y+dy, z+dz)
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
            # Add some randomness to avoid getting stuck in local minima of growth
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score descending, pick from top few or best
        options.sort(reverse=True)
        # With high bias, always pick best. With low bias, pick randomly from top 3.
        if bias > 0.8:
            site = options[0][1]
        else:
            top_k = min(3, len(options))
            site = options[rng.randint(0, top_k-1)][1]
            
        chain.append(site)
        occupied[site] = i
        
    return chain

def _tail_regrowth(sequence, chain, occupied, k, rng, temperature=1.0, h_set=None):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or None if stuck.
    Uses a greedy-with-randomness approach to choose the next step.
    """
    if h_set is None:
        h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    if k >= len(chain):
        return chain, occupied
    
    # Remove tail from k to end
    new_occupied = dict(occupied)
    for i in range(k, len(chain)):
        del new_occupied[chain[i]]
    
    new_chain = chain[:k]
    
    for i in range(k, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in new_occupied:
                continue
            
            score = 0
            if i in h_set:
                x, y, z = site
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (x+dx, y+dy, z+dz)
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 1
            # Add noise
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None
            
        # Boltzmann selection based on temperature
        if temperature < 0.1:
            # Greedy
            options.sort(reverse=True)
            site = options[0][1]
        else:
            # Softmax selection
            scores = [s for s, _ in options]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0
            site = options[0][1]
            for idx, (score, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break
        
        new_chain.append(site)
        new_occupied[site] = i
        
    return new_chain, new_occupied

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing for a single sequence.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
    
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    # Initialize with a few random greedy grows, pick the best
    best_init = None
    best_init_energy = float('inf')
    
    for _ in range(12):
        init_chain = _grow_greedy(sequence, dimension, rng, bias=0.4)
        if init_chain is None:
            continue
        init_occ = {p: i for i, p in enumerate(init_chain)}
        e = _energy_with_occupied(sequence, init_chain, init_occ, h_set)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain
            
    if best_init is None:
        # Fallback to a simple line
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)
        
    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Annealing schedule
    T = 3.0
    T_min = 0.01
    decay = 0.997
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random cut point k
        if rng.random() < 0.35:
            k = rng.randint(1, n - 1)
        else:
            # Biased towards end (shorter tails = faster moves)
            k = n - 1 - int(n * rng.expovariate(1.0 / (n * 0.07)))
            k = max(1, min(n - 1, k))
            
        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T, h_set)
        
        if new_chain is None:
            T *= decay
            continue
            
        new_energy = _energy_with_occupied(sequence, new_chain, new_occupied, h_set)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 1e-6 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= decay
        
    return best_chain

def _polish(sequence, chain, time_limit, rng):
    """
    Final polish pass with very low temperature to squeeze out remaining contacts.
    """
    n = len(sequence)
    if n < 2:
        return chain
    
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy_with_occupied(sequence, current_chain, current_occupied, h_set)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 0.05
    T_min = 0.001
    decay = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Focus on shorter tails for fine-tuning
        k = n - 1 - int(n * rng.expovariate(1.0 / (n * 0.04)))
        k = max(1, min(n - 1, k))
        
        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T, h_set)
        
        if new_chain is None:
            T *= decay
            continue
            
        new_energy = _energy_with_occupied(sequence, new_chain, new_occupied, h_set)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 1e-6 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= decay
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts with final polish.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Global deadline guard: 330s wall-clock, use 320s to be safe
    global_start = time.time()
    global_deadline = global_start + 320.0
    
    # Per-sequence time budget
    seq_time_limit = 11.0
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts
    num_restarts = 4
    slice_time = seq_time_limit / num_restarts
    
    rng = random.Random(hash(sequence) % (2**32))
    
    for restart in range(num_restarts):
        # Check global deadline
        elapsed = time.time() - global_start
        if elapsed > global_deadline - 5.0:  # 5s safety margin
            break
        
        # Run SA
        sa_chain = _sa_fold(sequence, dimension, rng, slice_time)
        
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain
                
    # Final polish pass with remaining time
    if best_chain is not None:
        elapsed = time.time() - global_start
        remaining = global_deadline - elapsed - 3.0  # 3s safety margin
        if remaining > 0.5:
            polished = _polish(sequence, best_chain, remaining, rng)
            pe = _energy(sequence, polished)
            if pe < best_energy:
                best_energy = pe
                best_chain = polished
    
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
