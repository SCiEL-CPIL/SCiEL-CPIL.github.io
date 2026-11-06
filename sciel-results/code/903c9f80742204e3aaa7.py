"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 180.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _calculate_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
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


def _pivot_move(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx.
    angle: 1 (90 deg), -1 (-90 deg), 2 (180 deg)
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n - 1:
        return None
    
    # Extract tail
    tail = chain[pivot_idx+1:]
    pivot_point = chain[pivot_idx]
    
    # Rotate tail around pivot_point
    new_tail = []
    occupied_head = set(chain[:pivot_idx+1])
    
    for p in tail:
        dx = p[0] - pivot_point[0]
        dy = p[1] - pivot_point[1]
        
        if angle == 1:
            ndx, ndy = -dy, dx
        elif angle == -1:
            ndx, ndy = dy, -dx
        else: # 180
            ndx, ndy = -dx, -dy
            
        new_p = (pivot_point[0] + ndx, pivot_point[1] + ndy)
        
        # Check collision with head
        if new_p in occupied_head:
            return None
        
        new_tail.append(new_p)
        
    return chain[:pivot_idx+1] + new_tail


def _optimize(sequence, chain, dimension, rng, time_limit):
    """Simulated Annealing with pivot moves."""
    if not chain or len(chain) < 3:
        return chain
        
    current_energy = _calculate_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    n = len(sequence)
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    start_t = time.time()
    
    while T > T_min and (time.time() - start_t) < time_limit:
        # Choose pivot
        pivot_idx = rng.randint(1, n - 2)
        angle = rng.choice([1, -1, 2])
        
        new_chain = _pivot_move(chain, pivot_idx, angle)
        
        if new_chain is None:
            T *= cooling
            continue
            
        new_energy = _calculate_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling
        
    return best_chain


def fold(sequence, dimension):
    """Best conformation using SA optimization."""
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time based on remaining global budget
    remaining = _time_left()
    if remaining <= 0:
        # Fallback
        return [list(p) for p in [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]]
        
    # Use a fraction of remaining time, but cap it to avoid starving later sequences
    # We have 10 sequences. Let's aim to use most of the budget.
    # If we assume equal distribution, each gets remaining/10.
    # But we want to be safe. Let's use a dynamic allocation.
    # For the first sequence, remaining is ~180. 180/10 = 18s.
    # Let's give it a bit more for the first few, or just a fixed fraction.
    # Let's try: time_limit = min(15.0, remaining * 0.15)
    # If remaining is 180, limit is 15.
    # If remaining is 50, limit is 7.5.
    time_limit = min(15.0, remaining * 0.15)
    
    # Ensure at least some time for short sequences
    if len(sequence) < 30:
        time_limit = min(time_limit, 5.0)
        
    best_chain = None
    best_energy = 0
    
    # Try a few initializations
    # More initializations for short sequences?
    n_restarts = 5
    
    for _ in range(n_restarts):
        if _time_left() <= 0:
            break
            
        init_chain = _grow(sequence, dimension, rng)
        
        if init_chain is None:
            continue
            
        # Optimize
        # Distribute time among restarts
        opt_chain = _optimize(sequence, init_chain, dimension, rng, time_limit / n_restarts)
        
        if opt_chain:
            e = _calculate_energy(sequence, opt_chain)
            if e < best_energy:
                best_energy = e
                best_chain = opt_chain

    if best_chain is None:
        # Fallback to straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
