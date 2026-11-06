"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # Budget is 120s total. Leave 10s safety margin.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
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

def _local_search(sequence, chain, rng, max_steps=2000):
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for _ in range(max_steps):
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        for new_pos in candidates:
            if new_pos in occupied:
                continue
            
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                del occupied[new_pos]
                occupied[curr] = i
                chain[i] = curr
    
    return chain, current_energy

def fold(sequence, dimension):
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    while _time_remaining() > 2.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        chain, energy = _local_search(sequence, chain, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
