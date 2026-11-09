"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_h_set(sequence):
    return frozenset(i for i, c in enumerate(sequence) if c == 'H')


def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    h_set = _build_h_set(sequence)
    if len(h_set) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in h_set:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _initial_grow(sequence, dimension, rng):
    """Greedy chain growth: prefer sites with new H-H contacts."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    h_set = _build_h_set(sequence)
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and occupied[q] in h_set)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _anneal(sequence, dimension, chain, rng, time_limit=5.0):
    """Simulated annealing to improve an initial chain."""
    if chain is None:
        return None, 0
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    h_set = _build_h_set(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T0 = 2.0
    T_end = 0.01
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        move_type = rng.random()
        
        if move_type < 0.5 and n > 5:
            # Move a single internal residue
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            prev_p = current_chain[i - 1]
            next_p = current_chain[i + 1]
            
            candidates = []
            for q in _neighbours(prev_p):
                if q == next_p:
                    continue
                if q in occupied and occupied[q] not in (i - 1, i + 1):
                    continue
                if q not in candidates:
                    candidates.append(q)
            for q in _neighbours(next_p):
                if q == prev_p:
                    continue
                if q in occupied and occupied[q] not in (i - 1, i + 1):
                    continue
                if q not in candidates:
                    candidates.append(q)
            
            if not candidates:
                continue
            
            new_site = rng.choice(candidates)
            if new_site == p:
                continue
            
            if new_site in occupied and occupied[new_site] not in (i - 1, i + 1):
                continue
            
            old_e = current_energy
            current_chain[i] = new_site
            occupied.pop(p, None)
            occupied[new_site] = i
            
            new_e = _energy(sequence, current_chain)
            delta_e = new_e - old_e
            
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / time_limit)
            T = T0 * (T_end / T0) ** progress
            
            if delta_e < 0 or rng.random() < min(1.0, math.exp(-delta_e / max(T, 1e-10))):
                current_energy = new_e
                if new_e < best_energy:
                    best_energy = new_e
                    best_chain = list(current_chain)
            else:
                current_chain[i] = p
                occupied.pop(new_site, None)
                occupied[p] = i
        
        else:
            # Move the last residue
            i = n - 1
            p = current_chain[i]
            prev_p = current_chain[i - 1]
            
            candidates = []
            for q in _neighbours(prev_p):
                if q in occupied and occupied[q] != i - 1:
                    continue
                if q not in candidates:
                    candidates.append(q)
            
            if not candidates:
                continue
            
            new_site = rng.choice(candidates)
            if new_site == p:
                continue
            
            if new_site in occupied and occupied[new_site] != i - 1:
                continue
            
            old_e = current_energy
            current_chain[i] = new_site
            occupied.pop(p, None)
            occupied[new_site] = i
            
            new_e = _energy(sequence, current_chain)
            delta_e = new_e - old_e
            
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / time_limit)
            T = T0 * (T_end / T0) ** progress
            
            if delta_e < 0 or rng.random() < min(1.0, math.exp(-delta_e / max(T, 1e-10))):
                current_energy = new_e
                if new_e < best_energy:
                    best_energy = new_e
                    best_chain = list(current_chain)
            else:
                current_chain[i] = p
                occupied.pop(new_site, None)
                occupied[p] = i
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    h_set = _build_h_set(sequence)
    
    if n <= 50:
        time_per_seq = 6.0
    elif n <= 80:
        time_per_seq = 8.0
    else:
        time_per_seq = 10.0
    
    rng = random.Random(12345 + len(sequence))
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    max_restarts = 10
    restarts_done = 0
    
    while restarts_done < max_restarts:
        if time.time() - start_time > time_per_seq * 0.85:
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            restarts_done += 1
            continue
        
        anneal_time = max(0.5, time_per_seq * 0.7 / (restarts_done + 1))
        chain, energy = _anneal(sequence, dimension, chain, rng, time_limit=anneal_time)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        restarts_done += 1
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
