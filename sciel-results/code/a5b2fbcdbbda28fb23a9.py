import random
import time
import math

_START_TIME = None
_TIME_BUDGET = 300.0

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return _START_TIME + _TIME_BUDGET - time.time()

def _neighbours(point):
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_energy(sequence, chain, coord_to_idx):
    contacts = 0
    n = len(sequence)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        pos = chain[i]
        for q in _neighbours(pos):
            if q in coord_to_idx:
                j = coord_to_idx[q]
                if j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _get_contact_count(sequence, pos, coord_to_idx, i):
    count = 0
    if sequence[i] != 'H':
        return 0
    for q in _neighbours(pos):
        if q in coord_to_idx:
            j = coord_to_idx[q]
            if abs(j - i) > 1 and sequence[j] == 'H':
                count += 1
    return count

def _grow(sequence, dimension, rng):
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    coord_to_idx = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in coord_to_idx:
                        j = coord_to_idx[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Add randomness to escape deterministic traps
            score = gain + rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None, None, None
            
        # Pick best
        candidates.sort(reverse=True)
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied.add(best_site)
        coord_to_idx[best_site] = i
        
    return chain, occupied, coord_to_idx

def _refine(sequence, dimension, rng, chain, occupied, coord_to_idx, max_steps=5000, temp_start=1.0, temp_end=0.01):
    n = len(sequence)
    current_energy = _calc_energy(sequence, chain, coord_to_idx)
    best_energy = current_energy
    best_chain = list(chain)
    best_occupied = set(occupied)
    best_coord_to_idx = dict(coord_to_idx)
    
    for step in range(max_steps):
        if _time_left() < 5:
            break
            
        progress = step / max_steps
        temp = temp_start * (temp_end / temp_start) ** progress
        
        i = rng.randint(1, n - 2)
        dir_vecs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        d = dir_vecs[rng.randint(0, 5)]
        
        current_pos = chain[i]
        new_pos = (current_pos[0] + d[0], current_pos[1] + d[1], current_pos[2] + d[2])
        
        if new_pos in occupied:
            continue
            
        old_contacts = _get_contact_count(sequence, current_pos, coord_to_idx, i)
        new_contacts = _get_contact_count(sequence, new_pos, coord_to_idx, i)
        
        delta_e = old_contacts - new_contacts # Energy change = -DeltaContacts
        
        if delta_e < 0 or rng.random() < min(1.0, math.exp(-delta_e / temp)):
            # Accept
            occupied.remove(current_pos)
            del coord_to_idx[current_pos]
            
            occupied.add(new_pos)
            coord_to_idx[new_pos] = i
            chain[i] = new_pos
            
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_occupied = set(occupied)
                best_coord_to_idx = dict(coord_to_idx)
        
    return best_chain, best_occupied, best_coord_to_idx, best_energy

def fold(sequence, dimension):
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    while _time_left() > 10:
        chain, occupied, coord_to_idx = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        if _time_left() < 5:
            break
            
        # Dynamic steps based on time
        # If we have lots of time, do more steps
        steps = 2000
        if len(sequence) > 50:
            steps = 5000
        if len(sequence) > 100:
            steps = 10000
            
        refined_chain, _, _, refined_energy = _refine(
            sequence, dimension, rng, chain, occupied, coord_to_idx, max_steps=steps
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
