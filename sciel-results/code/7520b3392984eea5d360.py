import random
import time

_DEADLINE = time.time() + 325

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

def _delta_energy(sequence, chain, i, p, q, where):
    res_i = sequence[i]
    if res_i != 'H':
        return 0
        
    delta = 0
    # Losses
    for n in _neighbours(p):
        j = where.get(n)
        if j is not None:
            if abs(j - i) > 1:
                if sequence[j] == 'H':
                    delta -= 1
                    
    # Gains
    for n in _neighbours(q):
        j = where.get(n)
        if j is not None:
            if abs(j - i) > 1:
                if sequence[j] == 'H':
                    delta += 1
                    
    return delta

def _refine(sequence, chain, deadline):
    n = len(chain)
    if n == 0:
        return chain
    
    current_energy = _energy(sequence, chain)
    occupied = set(chain)
    where = {p: i for i, p in enumerate(chain)}
    
    indices = [i for i in range(n) if sequence[i] == 'H']
    if not indices:
        return chain
        
    while time.time() < deadline:
        improved = False
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            p = chain[i]
            for q in _neighbours(p):
                if q in occupied:
                    continue
                
                d = _delta_energy(sequence, chain, i, p, q, where)
                
                if d < 0:
                    # Perform move
                    occupied.remove(p)
                    occupied.add(q)
                    where[p] = -1 # Mark as empty, or just delete? 
                    # Since where maps point->index, and p is now empty, we should remove it or set to -1.
                    # But q is now occupied by i.
                    del where[p]
                    where[q] = i
                    
                    chain[i] = q
                    current_energy += d
                    improved = True
                    break # Move to next i, as p changed? No, p is old pos. 
                    # Actually, after moving i, the neighborhood of i changes.
                    # So we should break the inner loop and move to next i.
        
        if not improved:
            break
            
    return chain

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    fallback = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    best_chain = fallback
    best_energy = _energy(sequence, fallback)
    
    while time.time() < _DEADLINE:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        refined_chain = _refine(sequence, chain, _DEADLINE)
        
        energy = _energy(sequence, refined_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = refined_chain
            
    return [list(p) for p in best_chain]
