# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _rotate_around_axis(v, axis, angle):
    """
    Rotate vector v (tuple of 3 ints) around axis (0, 1, or 2) by angle (1=90, 2=180, 3=270).
    """
    x, y, z = v
    if axis == 0: # Rotate around X
        if angle == 1: # 90 deg: (x, y, z) -> (x, -z, y)
            return (x, -z, y)
        elif angle == 2: # 180 deg: (x, y, z) -> (x, -y, -z)
            return (x, -y, -z)
        else: # 270 deg: (x, y, z) -> (x, z, -y)
            return (x, z, -y)
    elif axis == 1: # Rotate around Y
        if angle == 1: # (x, y, z) -> (z, y, -x)
            return (z, y, -x)
        elif angle == 2: # (x, y, z) -> (-x, y, -z)
            return (-x, y, -z)
        else: # (x, y, z) -> (-z, y, x)
            return (-z, y, x)
    else: # axis == 2, Rotate around Z
        if angle == 1: # (x, y, z) -> (-y, x, z)
            return (-y, x, z)
        elif angle == 2: # (x, y, z) -> (-x, -y, z)
            return (-x, -y, z)
        else: # (x, y, z) -> (y, -x, z)
            return (y, -x, z)


def _energy(sequence, chain):
    """Calculate energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _try_pivot(chain, sequence, dimension, i, angle):
    """
    Try to pivot the chain at index i (rotating tail i+1..N around bond i..i+1).
    Returns new chain if valid, else None.
    """
    if i < 0 or i >= len(chain) - 1:
        return None
    
    pivot_point = chain[i]
    bond_vector = (chain[i+1][0] - pivot_point[0], 
                   chain[i+1][1] - pivot_point[1], 
                   chain[i+1][2] - pivot_point[2])
    
    # Determine axis from bond_vector
    if bond_vector[0] != 0:
        axis = 0
    elif bond_vector[1] != 0:
        axis = 1
    else:
        axis = 2
        
    new_tail = []
    occupied_head = set(chain[:i+1]) # Points 0..i are fixed
    
    new_tail_set = set()
    valid = True
    
    for idx in range(i + 1, len(chain)):
        p = chain[idx]
        rel = (p[0] - pivot_point[0], p[1] - pivot_point[1], p[2] - pivot_point[2])
        new_rel = _rotate_around_axis(rel, axis, angle)
        new_p = (pivot_point[0] + new_rel[0], pivot_point[1] + new_rel[1], pivot_point[2] + new_rel[2])
        
        if new_p in occupied_head:
            valid = False
            break
        if new_p in new_tail_set:
            valid = False
            break
            
        new_tail.append(new_p)
        new_tail_set.add(new_p)
        
    if not valid:
        return None
        
    new_chain = chain[:i+1] + new_tail
    return new_chain


def _local_search(chain, sequence, dimension, rng, time_limit, start_time):
    """
    Hill climbing local search using pivot moves.
    """
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        n = len(current_chain)
        if n < 4:
            break
            
        # Pick random pivot index i (0 to n-2)
        i = rng.randint(0, n - 2)
        # Pick random angle (1, 2, 3)
        angle = rng.randint(1, 3)
        
        new_chain = _try_pivot(current_chain, sequence, dimension, i, angle)
        
        if new_chain is not None:
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                
    return current_chain, current_energy


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def fold(sequence, dimension):
    """
    Fold the sequence using iterative greedy growth and local search.
    """
    start_time = time.time()
    # Reserve 10 seconds for safety
    deadline = start_time + 280.0 
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1000 # Energy is negative, so 1000 is "infinity"
    
    while time.time() < deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        # Local search
        local_start = time.time()
        local_limit = min(0.5, deadline - time.time())
        if local_limit <= 0:
            break
            
        refined_chain, energy = _local_search(chain, sequence, dimension, rng, local_limit, local_start)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = refined_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
