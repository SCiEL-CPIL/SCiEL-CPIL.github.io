import time
import random
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate H-H contact energy change when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    x, y, z = q
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _compute_free_space(occupied, q):
    """Count free neighboring positions."""
    x, y, z = q
    free = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r not in occupied:
            free += 1
    return free

def _beam_search_fold(sequence, n, deadline, rng, beam_width=50):
    """
    Beam search: grow chain step by step, keeping only the best beam_width partial paths.
    Score = contacts * 10 + free_space * 1 - distance * 0.05
    """
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0.0)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, score in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                new_score = score
                if sequence[i] == 'H':
                    new_score += _calc_delta_energy(sequence, occupied, i, q) * 10.0
                new_score += _compute_free_space(occupied, q) * 1.0
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                new_score -= dist * 0.05
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, score in beams:
        if len(chain) == n:
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization: try single-residue moves to improve energy.
    """
    n = len(sequence)
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    def _get_energy():
        energy = 0
        for i in range(n):
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1 and sequence[j] == 'H':
                        energy -= 1
        return energy
    
    current_energy = _get_energy()
    improved = True
    max_iter = 500
    
    while improved and time.time() < deadline:
        improved = False
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            if n < 4:
                break
            i = rng.randint(1, n - 2)
            
            old_pos = chain[i]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if (q not in occupied or occupied[q] == i):
                    if i > 0:
                        prev = chain[i-1]
                        if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                            continue
                    if i < n-1:
                        nxt = chain[i+1]
                        if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                            continue
                    if q not in occupied or occupied[q] == i:
                        valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos:
                continue
            
            del occupied[old_pos]
            occupied[new_pos] = i
            old_chain_val = chain[i]
            chain[i] = new_pos
            
            new_energy = _get_energy()
            
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
            else:
                del occupied[new_pos]
                occupied[old_pos] = i
                chain[i] = old_chain_val
    
    return chain, current_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 220.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    beam_configs = [
        (50, 42), (80, 123), (120, 456), (60, 789), (100, 321),
        (70, 654), (90, 987), (40, 111), (150, 222), (65, 333)
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, rng_local, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        occupied = {}
        for idx, pos in enumerate(chain):
            occupied[pos] = idx
        
        energy = 0
        for i in range(n):
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1:
                        if sequence[j] == 'H':
                            energy -= 1
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        if n > 10 and best_energy <= -n * 0.45:
            break
    
    if best_chain is not None:
        for round_num in range(5):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
