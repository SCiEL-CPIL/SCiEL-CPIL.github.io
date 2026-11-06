"""HP lattice protein folding using Simulated Annealing with End and Pivot moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _neighbours(point):
    """Yield the 4 neighbors of a point."""
    x, y = point
    yield (x+1, y)
    yield (x-1, y)
    yield (x, y+1)
    yield (x, y-1)

def _compute_energy(sequence, chain):
    """Compute energy from scratch. O(N)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {tuple(p): i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy // 2

def _initial_spiral(n, rng):
    """Generate a spiral-like initial chain."""
    if n <= 1:
        return [(0, 0)] * n
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Directions: Right, Up, Left, Down
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d_idx = 0
    steps = 1
    while len(chain) < n:
        for _ in range(2):
            for _ in range(steps):
                if len(chain) >= n:
                    break
                dx, dy = dirs[d_idx]
                nx, ny = chain[-1][0] + dx, chain[-1][1] + dy
                if (nx, ny) not in occupied:
                    chain.append((nx, ny))
                    occupied.add((nx, ny))
            d_idx = (d_idx + 1) % 4
        steps += 1
    return chain

def _initial_random(n, rng):
    """Generate a random self-avoiding walk."""
    chain = [(0, 0)]
    occupied = {(0, 0)}
    for _ in range(n - 1):
        last = chain[-1]
        options = [p for p in _neighbours(last) if p not in occupied]
        if not options:
            return None
        next_p = rng.choice(options)
        chain.append(next_p)
        occupied.add(next_p)
    return chain

def _pivot_move(chain, rng):
    """
    Perform a pivot move.
    Returns new_chain or None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Pick pivot index p (1 to n-2)
    p = rng.randint(1, n - 2)
    # Pick rotation: 90, -90, 180
    rot = rng.choice([1, -1, 2]) # 1: 90 deg, -1: -90 deg, 2: 180 deg
    
    pivot_pt = chain[p]
    tail = chain[p+1:]
    
    new_tail = []
    # Rotate each point in tail around pivot
    for pt in tail:
        dx = pt[0] - pivot_pt[0]
        dy = pt[1] - pivot_pt[1]
        
        if rot == 1: # 90 deg CCW
            ndx, ndy = -dy, dx
        elif rot == -1: # 90 deg CW
            ndx, ndy = dy, -dx
        else: # 180 deg
            ndx, ndy = -dx, -dy
        
        new_tail.append((pivot_pt[0] + ndx, pivot_pt[1] + ndy))
    
    new_chain = chain[:p+1] + new_tail
    
    # Check self-avoidance
    seen = set()
    for pt in new_chain:
        if pt in seen:
            return None
        seen.add(pt)
        
    return new_chain

def _end_move(chain, rng):
    """
    Perform an end move (move last residue).
    Returns new_chain or None.
    """
    n = len(chain)
    if n < 2:
        return None
    
    last_pt = chain[-1]
    prev_pt = chain[-2]
    
    # Find empty neighbors of prev_pt that are not the last_pt
    options = []
    for q in _neighbours(prev_pt):
        if q != last_pt:
            # Check if q is occupied by any other part of the chain
            # For efficiency, we can just check if q is in the set of all points
            # But we don't have the set here. 
            # We can check if q is in chain[:-1]
            # This is O(N). 
            # To make it faster, we can pass a set, but let's keep it simple for now 
            # or optimize by checking against a set built once? 
            # No, we are in a move proposal. 
            # Let's assume N is small enough.
            
            # Optimization: check if q is in the chain
            # We can use a set if we maintain one, but we are returning a new chain.
            # Let's just check.
            is_occupied = False
            for pt in chain[:-1]:
                if pt == q:
                    is_occupied = True
                    break
            if not is_occupied:
                options.append(q)
                
    if not options:
        return None
        
    new_last = rng.choice(options)
    new_chain = chain[:-1] + [new_last]
    return new_chain

def _sa_fold(sequence, n, dimension, rng, time_budget):
    """Run SA with End and Pivot moves."""
    start_time = time.time()
    
    # Initialize
    chain = _initial_spiral(n, rng)
    if chain is None or len(chain) != n:
        chain = _initial_random(n, rng)
    if chain is None:
        return None, 0
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = [list(p) for p in chain]
    best_energy = current_energy
    
    T = 2.0
    T_min = 0.05
    cooling_rate = 0.99
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
            
        # Choose move
        if rng.random() < 0.5:
            new_chain = _end_move(chain, rng)
        else:
            new_chain = _pivot_move(chain, rng)
            
        if new_chain is None:
            T *= cooling_rate
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in chain]
                
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using multiple SA restarts with End and Pivot moves.
    """
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
        
    total_time_budget = 110.0 # Leave some margin for 120s limit
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    # Determine number of restarts
    # Let's do 10 restarts, each with 10% of time
    n_restarts = 10
    time_per_restart = total_time_budget / n_restarts
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = total_time_budget - elapsed
        if remaining < 0.1:
            break
            
        # Create a new RNG for each restart to ensure diversity
        # Seed based on sequence and restart index
        seed = hash(sequence) + i * 1000
        rng = random.Random(seed)
        
        # Use slightly less time than allocated to account for overhead
        alloc_time = min(time_per_restart, remaining)
        
        chain, energy = _sa_fold(sequence, n, dimension, rng, alloc_time)
        
        if chain is not None:
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
                
    if best_chain is None:
        # Fallback
        best_chain = [[i, 0] for i in range(n)]
        
    return best_chain
# EVOLVE-BLOCK-END
