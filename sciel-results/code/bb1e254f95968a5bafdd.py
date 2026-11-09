"""
HP Lattice Protein Folding using Global Regrowth Search with Simulated Annealing.
Structural change: Replace local pivot/crankshaft moves with full-chain regrowth
and multi-scale segment regrowth for better exploration of conformational space.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _compute_energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_chain(sequence, chain_prefix, occupied_prefix, rng, temperature=0.0):
    """
    Grow the chain from the end of chain_prefix to complete the full sequence.
    Uses Boltzmann sampling at given temperature.
    Returns (complete_chain, complete_occupied) or (None, None) if stuck.
    """
    n = len(sequence)
    current_chain = list(chain_prefix)
    current_occupied = dict(occupied_prefix)
    
    for i in range(len(chain_prefix), n):
        last = current_chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in current_occupied:
                continue
            
            # Score based on potential H-H contacts
            score = 0.0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in current_occupied:
                        j = current_occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 1.0
            
            # Add small random perturbation
            score += rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None, None
        
        # Select site using Boltzmann distribution
        if temperature < 0.01:
            # Greedy: pick best
            options.sort(reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0.0
            site = options[0][1]
            for idx, (score, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break
        
        current_chain.append(site)
        current_occupied[site] = i
    
    return current_chain, current_occupied

def _global_regrowth_sa(sequence, rng, time_limit):
    """
    Simulated annealing with global regrowth moves.
    At each step, pick a random cut point and regrow the entire tail.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
    
    # Initialize with greedy growth
    best_init = None
    best_init_energy = float('inf')
    
    for _ in range(20):
        init_chain, init_occupied = _grow_chain(sequence, [(0, 0, 0)], {(0, 0, 0): 0}, rng, temperature=0.5)
        if init_chain is None:
            continue
        e = _compute_energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain
    
    if best_init is None:
        # Fallback: straight line
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _compute_energy(sequence, best_init)
    
    current_chain = list(best_init)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Adaptive temperature schedule
    T = 5.0
    T_min = 0.05
    cooling_rate = 0.995
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Choose cut point: bias towards middle/end for larger moves
        r = rng.random()
        if r < 0.3:
            # Short tail (last 5-15 residues)
            tail_len = rng.randint(5, min(15, n - 1))
            k = max(1, n - tail_len)
        elif r < 0.7:
            # Medium tail (20-50% of chain)
            tail_frac = rng.uniform(0.2, 0.5)
            k = max(1, int(n * (1.0 - tail_frac)))
        else:
            # Long tail (50-90% of chain) - more global moves
            tail_frac = rng.uniform(0.5, 0.9)
            k = max(1, int(n * (1.0 - tail_frac)))
        
        # Regrow from cut point
        prefix = current_chain[:k]
        prefix_occupied = {p: i for i, p in enumerate(prefix)}
        
        new_chain, new_occupied = _grow_chain(sequence, prefix, prefix_occupied, rng, T)
        
        if new_chain is None:
            T *= cooling_rate
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e < 0 or (T > 0.01 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling_rate
    
    return best_chain

def _multi_scale_regrowth(sequence, rng, time_limit):
    """
    Multi-scale regrowth: alternate between local, medium, and global moves.
    Uses a more sophisticated temperature schedule.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
    
    # Multiple initializations
    best_init = None
    best_init_energy = float('inf')
    
    for _ in range(25):
        # Vary temperature for diversity
        temp = rng.uniform(0.1, 2.0)
        init_chain, _ = _grow_chain(sequence, [(0, 0, 0)], {(0, 0, 0): 0}, rng, temperature=temp)
        if init_chain is None:
            continue
        e = _compute_energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain
    
    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _compute_energy(sequence, best_init)
    
    current_chain = list(best_init)
    current_energy = best_init_energy
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Phase 1: Global exploration (high T, large moves)
    # Phase 2: Medium refinement (medium T, medium moves)
    # Phase 3: Local polishing (low T, small moves)
    
    phase1_end = deadline - (deadline - start_time) * 0.6
    phase2_end = deadline - (deadline - start_time) * 0.2
    
    T = 8.0
    T_min = 0.02
    cooling_rate = 0.993
    
    while time.time() < deadline:
        # Determine phase
        if time.time() < phase1_end:
            # Global phase
            move_scale = 'global'
            T = max(T_min, 5.0 * math.exp(-0.001 * (time.time() - start_time)))
        elif time.time() < phase2_end:
            # Medium phase
            move_scale = 'medium'
            T = max(T_min, 2.0 * math.exp(-0.002 * (time.time() - phase1_end)))
        else:
            # Local phase
            move_scale = 'local'
            T = max(T_min, 0.5 * math.exp(-0.005 * (time.time() - phase2_end)))
        
        # Choose cut point based on phase
        if move_scale == 'global':
            r = rng.random()
            if r < 0.5:
                k = rng.randint(1, max(1, n // 2))
            else:
                k = rng.randint(n // 2, max(1, n - 1))
        elif move_scale == 'medium':
            tail_len = rng.randint(10, min(30, n - 1))
            k = max(1, n - tail_len)
        else:
            tail_len = rng.randint(3, min(10, n - 1))
            k = max(1, n - tail_len)
        
        # Regrow
        prefix = current_chain[:k]
        prefix_occupied = {p: i for i, p in enumerate(prefix)}
        
        new_chain, _ = _grow_chain(sequence, prefix, prefix_occupied, rng, T)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e < 0 or (T > 0.01 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using multi-scale regrowth SA with multiple restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Time budget: 26 sequences in 330s total, leave safety margin
    # Use adaptive time allocation based on chain length
    if n <= 50:
        seq_time_limit = 11.0
    elif n <= 80:
        seq_time_limit = 13.0
    else:
        seq_time_limit = 15.0
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts with different strategies
    if n <= 50:
        num_restarts = 5
    elif n <= 80:
        num_restarts = 4
    else:
        num_restarts = 3
    
    slice_time = seq_time_limit / num_restarts
    
    rng = random.Random(hash(sequence) % (2**32))
    
    # Alternate between different search strategies
    for restart in range(num_restarts):
        if time.time() > 325.0:  # Safety margin
            break
        
        if restart % 2 == 0:
            sa_chain = _global_regrowth_sa(sequence, rng, slice_time)
        else:
            sa_chain = _multi_scale_regrowth(sequence, rng, slice_time)
        
        if sa_chain is not None:
            e = _compute_energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
