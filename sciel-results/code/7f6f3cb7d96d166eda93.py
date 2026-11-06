"""Improved HP lattice folding with high-volume randomized growth and time management."""
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
    # Leave 10s safety margin before 120s budget (or 210s wall-clock limit)
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
    This is the core strategy from the high-performing inspiration.
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
            # The random component ensures diversity across restarts
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

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses a high number of randomized greedy growths to find low-energy conformations.
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
    
    # Perform many restarts. The growth is fast, so we can do many.
    # We loop until time runs out, but cap iterations to avoid infinite loops if time is mocked.
    max_iterations = 5000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Early termination if we find a very good solution?
        # For standard HP benchmarks, optimal energy is often around -n/2 to -n.
        # We don't know the exact optimum, so we rely on the high number of restarts.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
