"""Improved HP lattice protein folding with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU. Safety margin to 110s.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then random tie-breaker
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, max_steps=500):
    """
    Refine the chain using kink moves and segment moves with simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    is_h = [c == 'H' for c in sequence]
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.05
    
    for step in range(max_steps):
        # Annealing schedule
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type: 50% kink, 50% segment
        move_type = rng.random()
        
        if move_type < 0.5:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            # Find valid new positions for curr
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
                if dist == 1:
                    candidates.add(p)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(list(candidates))
            
            if new_pos in occupied:
                continue
            
            # Try the move
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
            else:
                # Revert
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
        else:
            # Segment move (2-mer kink)
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            p0 = chain[i-1]
            p1 = chain[i]
            p2 = chain[i+1]
            p3 = chain[i+2]
            
            found = False
            for p1_new in _neighbours(p0):
                if p1_new == p2:
                    continue
                for p2_new in _neighbours(p1_new):
                    if p2_new == p0:
                        continue
                    if not (abs(p2_new[0] - p3[0]) + abs(p2_new[1] - p3[1]) == 1):
                        continue
                    
                    # Check self-avoidance
                    if p1_new in occupied or p2_new in occupied:
                        continue
                    
                    # Try the move
                    del occupied[p1]
                    del occupied[p2]
                    occupied[p1_new] = i
                    occupied[p2_new] = i + 1
                    chain[i] = p1_new
                    chain[i+1] = p2_new
                    
                    new_energy = _energy(sequence, chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                        current_energy = new_energy
                    else:
                        # Revert
                        chain[i] = p1
                        chain[i+1] = p2
                        del occupied[p1_new]
                        del occupied[p2_new]
                        occupied[p1] = i
                        occupied[p2] = i + 1
                    found = True
                    break
                if found:
                    break
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses randomized greedy growth followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we look for lower (more negative) values
    
    # Perform many restarts
    max_iterations = 5000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Local Search
        # Adjust local search steps based on remaining time and sequence length
        # Shorter sequences can afford more steps, or we just keep it fixed.
        # Let's use 100 steps for local search to keep it fast.
        chain, energy = _local_search(sequence, chain, rng, max_steps=100)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            # If we find a very good solution, we might want to keep searching
            # to see if we can do better, but we are already in a loop.
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
