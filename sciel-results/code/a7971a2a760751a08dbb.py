import random
import math
import time

def _neighbours(point):
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
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
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _apply_pivot(chain, cut_idx, axis, angle):
    pivot_point = chain[cut_idx - 1]
    segment = chain[cut_idx:]
    
    def rotate_point(p, axis, angle):
        x, y, z = p
        if axis == 0:
            if angle == 1: return (0, -z, y)
            elif angle == 2: return (0, -y, -z)
            else: return (0, z, -y)
        elif axis == 1:
            if angle == 1: return (z, 0, -x)
            elif angle == 2: return (-x, 0, -z)
            else: return (-z, 0, x)
        else:
            if angle == 1: return (-y, x, 0)
            elif angle == 2: return (-x, -y, 0)
            else: return (y, -x, 0)

    rel_segment = [tuple(p[k] - pivot_point[k] for k in range(3)) for p in segment]
    new_rel_segment = [rotate_point(p, axis, angle) for p in rel_segment]
    new_segment = [tuple(p[k] + pivot_point[k] for k in range(3)) for p in new_rel_segment]
    
    fixed_set = set(chain[:cut_idx])
    for p in new_segment:
        if p in fixed_set:
            return None
            
    return chain[:cut_idx] + new_segment

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    
    start_time = time.time()
    time_limit = 10.0
    
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    current_energy = _energy(sequence, chain)
    best_chain = chain
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        move_type = rng.random()
        if move_type < 0.5:
            last_idx = len(chain) - 1
            neighbors = list(_neighbours(chain[-1]))
            occupied = set(chain)
            valid_neighbors = [n for n in neighbors if n not in occupied]
            if valid_neighbors:
                new_pos = rng.choice(valid_neighbors)
                new_chain = chain[:-1] + [new_pos]
                new_energy = _energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain
        else:
            if len(chain) > 2:
                cut_idx = rng.randint(1, len(chain) - 1)
                axis = rng.randint(0, 2)
                angle = rng.randint(1, 3)
                new_chain = _apply_pivot(chain, cut_idx, axis, angle)
                if new_chain is not None:
                    new_energy = _energy(sequence, new_chain)
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = chain
        
        T *= cooling
        if T < 0.01:
            break
            
    return [list(p) for p in best_chain]
