"""Improved HP lattice protein folding using Simulated Annealing with Suffix Regrowth."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_chain(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using greedy growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Prefer sites that create H-H contacts
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _regrow_suffix(sequence, dimension, chain, start_idx, rng, max_backtrack=20):
    """
    Attempt to regrow the chain from start_idx to the end.
    Returns the new suffix (list of points) or None if failed.
    Uses a randomized greedy approach with limited backtracking.
    """
    n = len(sequence)
    if start_idx >= n:
        return []
    
    # We keep chain[0..start_idx-1] fixed.
    # We need to generate chain[start_idx..n-1].
    # Current last point is chain[start_idx-1].
    
    # We will use a simple recursive backtracking with a depth limit or just a greedy random walk.
    # Given time constraints, a simple greedy random walk with restarts is safer.
    # Let's try a few random walks.
    
    last = chain[start_idx - 1]
    occupied_prefix = set(chain[:start_idx])
    
    # To speed up, we can just try a few random greedy paths
    for attempt in range(10):
        current_chain = [last]
        current_occupied = set(occupied_prefix)
        success = True
        for i in range(start_idx, n):
            options = []
            for site in _neighbours(current_chain[-1]):
                if site in current_occupied:
                    continue
                options.append(site)
            
            if not options:
                success = False
                break
            
            # Choose randomly, maybe biased towards compactness
            site = rng.choice(options)
            current_chain.append(site)
            current_occupied.add(site)
        
        if success and len(current_chain) == n - start_idx:
            return current_chain
    
    return None

def _calc_energy_change(sequence, chain, start_idx, new_suffix):
    """
    Calculate the change in energy when replacing chain[start_idx:] with new_suffix.
    Returns delta_energy (new_energy - old_energy).
    """
    n = len(sequence)
    old_suffix = chain[start_idx:]
    
    # We need to count contacts involving indices >= start_idx.
    # Contacts are between i and j where |i-j| > 1 and dist(i,j)=1.
    
    # Build set of occupied points for the whole chain to check neighbors easily?
    # It's better to just recount the energy for the whole chain if it's fast enough.
    # For N=136, N^2 is ~18k, which is very fast.
    # Let's just compute full energy for simplicity and correctness, as local optimization might be tricky to debug.
    # But we call this many times.
    # Let's optimize: 
    # 1. Remove contacts from old suffix.
    # 2. Add contacts from new suffix.
    
    # This is complex. Let's stick to full energy calculation but only if we accept the move?
    # No, we need energy to decide acceptance.
    # Full energy calculation is O(N). With N=136, it's trivial.
    # Let's just compute full energy.
    
    new_chain = chain[:start_idx] + new_suffix
    return _energy(sequence, new_chain)

def fold(sequence, dimension):
    """
    Best conformation using Simulated Annealing with Suffix Regrowth.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Deadline guard
    start_time = time.time()
    deadline = 320.0  # Leave 10s margin for 330s limit
    
    rng = random.Random(12345 + len(sequence))
    
    # 1. Initial Solution
    best_chain = _initial_chain(sequence, dimension, rng)
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_energy = _energy(sequence, best_chain)
    current_chain = best_chain
    current_energy = best_energy
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Main Loop
    while time.time() < deadline:
        # Pick a random index to cut
        # Bias towards the end? Or uniform? Uniform is good for exploration.
        # Cutting at 0 means regrowing whole chain (expensive).
        # Let's pick i from 1 to n-1.
        cut_idx = rng.randint(1, n - 1)
        
        # Regrow suffix
        new_suffix = _regrow_suffix(sequence, dimension, current_chain, cut_idx, rng)
        
        if new_suffix is None:
            # Failed to regrow, maybe try a smaller cut or just continue
            # To avoid getting stuck in a state where we can't regrow,
            # we can occasionally try a "creep" or just accept the failure.
            # For now, just continue.
            continue
        
        new_chain = current_chain[:cut_idx] + new_suffix
        new_energy = _energy(sequence, new_chain)
        
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
