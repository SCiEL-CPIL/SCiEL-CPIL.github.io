import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng):
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.5 * free + rng.random(), site))
        
        if not options:
            return None
            
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied.add(best_site)
        
    return chain

def fold(sequence, dimension):
    N = len(sequence)
    if N <= 1:
        return [[0] * dimension]
    
    global_start = time.time()
    deadline = global_start + 10.0
    
    best_chain = None
    best_energy = 0
    
    is_h = [c == 'H' for c in sequence]
    
    dirs = []
    for axis in range(dimension):
        for step in (-1, 1):
            d = [0] * dimension
            d[axis] = step
            dirs.append(tuple(d))
            
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
            
        rng = random.Random(12345 + N + restart)
        
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            chain = [(i, 0, 0) for i in range(N)]
            
        chain = [tuple(p) for p in chain]
        current_energy = _compute_energy(sequence, chain)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain[:]
            
        # Local Search
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        steps = 5000
        
        occupied = set(chain)
        pos_to_idx = {p: i for i, p in enumerate(chain)}
        
        def get_delta_energy(old_pos, new_pos, idx):
            delta = 0
            if is_h[idx]:
                for q in _neighbours(old_pos):
                    j = pos_to_idx.get(q)
                    if j is not None and j != idx and abs(j - idx) > 1 and is_h[j]:
                        delta -= 1
                for q in _neighbours(new_pos):
                    j = pos_to_idx.get(q)
                    if j is not None and j != idx and abs(j - idx) > 1 and is_h[j]:
                        delta += 1
            return delta

        for step in range(steps):
            if time.time() > deadline - 0.5:
                break
                
            # Kink Move
            if N < 3:
                break
            i = rng.randint(1, N - 2)
            p = chain[i]
            d = rng.choice(dirs)
            new_p = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            
            if new_p in occupied:
                continue
                
            delta_e = get_delta_energy(p, new_p, i)
            
            if delta_e <= 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
                occupied.remove(p)
                occupied.add(new_p)
                pos_to_idx.pop(p)
                pos_to_idx[new_p] = i
                chain[i] = new_p
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    
            T *= cooling
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(N)]
        
    return [list(p) for p in best_chain]
