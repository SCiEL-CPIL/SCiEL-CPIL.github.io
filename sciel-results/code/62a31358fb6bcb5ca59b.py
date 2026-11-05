"""HP lattice protein folding using Simulated Annealing with diverse moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.92  # Use 92% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer positions with more H-H contacts and more free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _simulated_annealing(sequence, chain, dimension, rng, time_limit=1.0):
    """Improve a chain using Simulated Annealing."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map
    pos_map = {p: i for i, p in enumerate(chain)}
    
    def compute_energy():
        contacts = 0
        for i, p in enumerate(chain):
            if sequence[i] == "H":
                for q in _neighbours(p):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
        return -contacts
    
    current_energy = compute_energy()
    best_energy = current_energy
    best_chain = list(chain)
    
    # Precompute which indices are H
    is_h = [1 if s == "H" else 0 for s in sequence]
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    iterations = 5000
    alpha = (T_end / T_start) ** (1.0 / iterations)
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(iterations):
        if _is_deadline_exceeded() or (time.time() - start_time > time_limit):
            break
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        
        p_i = chain[i]
        
        # Try all 6 neighbours
        for site in _neighbours(p_i):
            # Check if site is valid
            if site in pos_map:
                continue
            if i > 0 and site == chain[i-1]:
                continue
            if i < n-1 and site == chain[i+1]:
                continue
            
            # Check if moving p_i to site is valid
            # Must be adjacent to chain[i-1] and chain[i+1]
            valid = True
            if i > 0:
                if abs(site[0]-chain[i-1][0]) + abs(site[1]-chain[i-1][1]) + abs(site[2]-chain[i-1][2]) != 1:
                    valid = False
            if valid and i < n-1:
                if abs(site[0]-chain[i+1][0]) + abs(site[1]-chain[i+1][1]) + abs(site[2]-chain[i+1][2]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Try the move
            old_site = chain[i]
            chain[i] = site
            del pos_map[old_site]
            pos_map[site] = i
            
            new_energy = compute_energy()
            delta = new_energy - current_energy
            
            # Acceptance criterion
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_site
                del pos_map[site]
                pos_map[old_site] = i
            
            # Update temperature
            T *= alpha
            break  # Only one move per iteration
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multi-start greedy growth and simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # Multiple restarts with greedy growth + simulated annealing
    for restart in range(20):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply simulated annealing to improve
        chain = _simulated_annealing(sequence, chain, dimension, rng, time_limit=10.0)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
