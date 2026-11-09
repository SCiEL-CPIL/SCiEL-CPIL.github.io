"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

START_TIME = time.time()
DEADLINE = START_TIME + 300.0

def _neighbours(point):
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

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

def _greedy_init(sequence, dimension, rng):
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 5.0 + free * 0.1 + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        best_site = max(options)[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _simulate_annealing(sequence, dimension, rng, time_limit):
    N = len(sequence)
    if N < 2:
        return [(0,0,0)]
    
    chain = _greedy_init(sequence, dimension, rng)
    if chain is None:
        return [(0,0,0)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    T_start = 2.0
    T_end = 0.01
    steps = 100000
    
    for step in range(steps):
        if time.time() > time_limit:
            break
            
        if rng.random() < 0.5:
            # Kink
            if N < 3: continue
            i = rng.randint(1, N-2)
            j = i + 1
            axis = rng.randint(0, 2)
            sign = 1 if rng.random() < 0.5 else -1
            
            pivot = chain[i-1]
            p_i = chain[i]
            p_j = chain[j]
            
            del occupied[p_i]
            del occupied[p_j]
            
            def rotate(p, pivot, axis, sign):
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                dz = p[2] - pivot[2]
                if axis == 0:
                    if sign == 1: return (pivot[0]+dx, pivot[1]-dz, pivot[2]+dy)
                    else: return (pivot[0]+dx, pivot[1]+dz, pivot[2]-dy)
                elif axis == 1:
                    if sign == 1: return (pivot[0]+dz, pivot[1]+dy, pivot[2]-dx)
                    else: return (pivot[0]-dz, pivot[1]+dy, pivot[2]+dx)
                else:
                    if sign == 1: return (pivot[0]+dy, pivot[1]-dx, pivot[2]+dz)
                    else: return (pivot[0]-dy, pivot[1]+dx, pivot[2]-dz)
            
            new_p_i = rotate(p_i, pivot, axis, sign)
            new_p_j = rotate(p_j, pivot, axis, sign)
            
            valid = True
            if new_p_i in occupied: valid = False
            if new_p_j in occupied: valid = False
            
            if valid:
                old_c = 0
                if sequence[i] == "H":
                    for nb in _neighbours(p_i):
                        k = occupied.get(nb)
                        if k is not None and abs(i-k)>1 and sequence[k]=="H":
                            old_c += 1
                if sequence[j] == "H":
                    for nb in _neighbours(p_j):
                        k = occupied.get(nb)
                        if k is not None and abs(j-k)>1 and sequence[k]=="H":
                            old_c += 1
                            
                chain[i] = new_p_i
                chain[j] = new_p_j
                occupied[new_p_i] = i
                occupied[new_p_j] = j
                
                new_c = 0
                if sequence[i] == "H":
                    for nb in _neighbours(new_p_i):
                        k = occupied.get(nb)
                        if k is not None and abs(i-k)>1 and sequence[k]=="H":
                            new_c += 1
                if sequence[j] == "H":
                    for nb in _neighbours(new_p_j):
                        k = occupied.get(nb)
                        if k is not None and abs(j-k)>1 and sequence[k]=="H":
                            new_c += 1
                            
                delta_e = old_c - new_c
                
                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    chain[i] = p_i
                    chain[j] = p_j
                    occupied[p_i] = i
                    occupied[p_j] = j
                    del occupied[new_p_i]
                    del occupied[new_p_j]
            else:
                occupied[p_i] = i
                occupied[p_j] = j
        else:
            # Pivot
            if N < 3: continue
            i = rng.randint(1, N-1)
            max_len = N - i
            L = rng.randint(2, max_len)
            j = i + L - 1
            if j >= N: continue
            
            axis = rng.randint(0, 2)
            sign = 1 if rng.random() < 0.5 else -1
            
            pivot = chain[i-1]
            seg = chain[i:j+1]
            
            for p in seg:
                del occupied[p]
                
            new_seg = []
            valid = True
            for p in seg:
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                dz = p[2] - pivot[2]
                if axis == 0:
                    if sign == 1: np_ = (pivot[0]+dx, pivot[1]-dz, pivot[2]+dy)
                    else: np_ = (pivot[0]+dx, pivot[1]+dz, pivot[2]-dy)
                elif axis == 1:
                    if sign == 1: np_ = (pivot[0]+dz, pivot[1]+dy, pivot[2]-dx)
                    else: np_ = (pivot[0]-dz, pivot[1]+dy, pivot[2]+dx)
                else:
                    if sign == 1: np_ = (pivot[0]+dy, pivot[1]-dx, pivot[2]+dz)
                    else: np_ = (pivot[0]-dy, pivot[1]+dx, pivot[2]-dz)
                    
                if np_ in occupied:
                    valid = False
                    break
                new_seg.append(np_)
                
            if not valid:
                for k, p in enumerate(seg):
                    occupied[p] = i+k
                continue
                
            old_c = 0
            for idx in range(i, j+1):
                if sequence[idx] != "H": continue
                pos = chain[idx]
                for nb in _neighbours(pos):
                    k = occupied.get(nb)
                    if k is not None and abs(idx - k) > 1 and sequence[k] == "H":
                         old_c += 1
                         
            for k, p in enumerate(new_seg):
                chain[i+k] = p
                occupied[p] = i+k
                 
            new_c = 0
            for idx in range(i, j+1):
                if sequence[idx] != "H": continue
                pos = chain[idx]
                for nb in _neighbours(pos):
                    k = occupied.get(nb)
                    if k is not None and abs(idx - k) > 1 and sequence[k] == "H":
                         new_c += 1
                         
            delta_e = old_c - new_c
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                for k, p in enumerate(seg):
                    chain[i+k] = p
                    occupied[p] = i+k
                for p in new_seg:
                    del occupied[p]
                    
        T = T_start * (T_end / T_start) ** (step / steps)
        
    return best_chain

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    
    remaining = DEADLINE - time.time()
    if remaining < 1.0:
        return [(i, 0, 0) for i in range(len(sequence))]
        
    # Assume ~26 sequences. 240s / 26 ~ 9s. Use 8s to be safe.
    time_limit = time.time() + min(8.0, remaining * 0.9)
    
    best_chain = _simulate_annealing(sequence, dimension, rng, time_limit)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
