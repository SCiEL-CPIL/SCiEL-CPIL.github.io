"""HP lattice protein folding on the cubic lattice - SA with Pivots and simple energy."""
import time
import random
import math

# EVOLVE-BLOCK-START
import time as _time

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _init_chain(sequence, rng):
    """Initialize a self-avoiding walk with backtracking."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    for attempt in range(50):
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0)}
        success = True
        for i in range(1, n):
            last = chain[-1]
            moves = _NEIGHBOR_OFFSETS[:]
            rng.shuffle(moves)
            placed = False
            for d in moves:
                new_pt = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
                if new_pt not in occupied:
                    chain.append(new_pt)
                    occupied.add(new_pt)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return chain
    
    return [(i, 0, 0) for i in range(n)]

def _energy(sequence, chain):
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    energy = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        pi = chain[i]
        for d in _NEIGHBOR_OFFSETS:
            neighbor = (pi[0] + d[0], pi[1] + d[1], pi[2] + d[2])
            j = pos_to_idx.get(neighbor)
            if j is not None and j > i + 1:
                energy -= 1
    return energy

def _pivot_mc_fold(sequence, n_restarts, time_budget, rng):
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = _time.time()
    total_deadline = start_time + time_budget
    
    for restart in range(n_restarts):
        if _time.time() > total_deadline - 2:
            break
        
        chain = _init_chain(sequence, rng)
        if len(chain) < n:
            chain = [(i, 0, 0) for i in range(n)]
            
        current_energy = _energy(sequence, chain)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
        
        T_start = max(1.0, abs(current_energy) * 0.5)
        T_end = 0.05
        n_steps = max(1000, min(5000, n * 50))
        
        for step in range(n_steps):
            if _time.time() > total_deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            k = rng.randint(1, n - 2)
            rot_type = rng.randint(0, 5)
            
            pivot_point = chain[k]
            old_tail = chain[k+1:]
            new_tail = []
            valid = True
            head_set = set(chain[:k+1])
            
            for pt in old_tail:
                rx, ry, rz = pt[0] - pivot_point[0], pt[1] - pivot_point[1], pt[2] - pivot_point[2]
                if rot_type == 0:
                    nx, ny, nz = rx, -rz, ry
                elif rot_type == 1:
                    nx, ny, nz = rx, rz, -ry
                elif rot_type == 2:
                    nx, ny, nz = rz, ry, -rx
                elif rot_type == 3:
                    nx, ny, nz = -rz, ry, rx
                elif rot_type == 4:
                    nx, ny, nz = -ry, rx, rz
                else:
                    nx, ny, nz = ry, -rx, rz
                    
                new_pt = (pivot_point[0] + nx, pivot_point[1] + ny, pivot_point[2] + nz)
                if new_pt in head_set:
                    valid = False
                    break
                new_tail.append(new_pt)
            
            if valid:
                tail_set = set()
                for pt in new_tail:
                    if pt in tail_set:
                        valid = False
                        break
                    tail_set.add(pt)
            
            if not valid:
                continue
                
            new_chain = chain[:k+1] + new_tail
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    
        # Local improvement
        improved = True
        while improved and _time.time() < total_deadline - 5:
            improved = False
            for k in range(1, n - 1):
                for rot_type in range(6):
                    pivot_point = chain[k]
                    old_tail = chain[k+1:]
                    new_tail = []
                    valid = True
                    head_set = set(chain[:k+1])
                    
                    for pt in old_tail:
                        rx, ry, rz = pt[0] - pivot_point[0], pt[1] - pivot_point[1], pt[2] - pivot_point[2]
                        if rot_type == 0:
                            nx, ny, nz = rx, -rz, ry
                        elif rot_type == 1:
                            nx, ny, nz = rx, rz, -ry
                        elif rot_type == 2:
                            nx, ny, nz = rz, ry, -rx
                        elif rot_type == 3:
                            nx, ny, nz = -rz, ry, rx
                        elif rot_type == 4:
                            nx, ny, nz = -ry, rx, rz
                        else:
                            nx, ny, nz = ry, -rx, rz
                            
                        new_pt = (pivot_point[0] + nx, pivot_point[1] + ny, pivot_point[2] + nz)
                        if new_pt in head_set:
                            valid = False
                            break
                        new_tail.append(new_pt)
                    
                    if valid:
                        tail_set = set()
                        for pt in new_tail:
                            if pt in tail_set:
                                valid = False
                                break
                            tail_set.add(pt)
                    
                    if not valid:
                        continue
                        
                    new_chain = chain[:k+1] + new_tail
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        chain = new_chain
                        current_energy = new_energy
                        improved = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            
    return best_chain if best_chain else [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    if not hasattr(fold, '_start_time'):
        fold._start_time = _time.time()
        
    elapsed = _time.time() - fold._start_time
    per_call_budget = min(20.0, max(2.0, 280.0 - elapsed))
    
    if n <= 48:
        n_restarts = max(2, int(per_call_budget / 2.0))
    elif n <= 80:
        n_restarts = max(1, int(per_call_budget / 4.0))
    else:
        n_restarts = max(1, int(per_call_budget / 6.0))
        
    rng = random.Random(42 + n * 7)
    
    best_chain = _pivot_mc_fold(sequence, n_restarts, per_call_budget, rng)
    
    if len(set(best_chain)) != n:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = [(i, 0, 0) for i in range(n)]
            break
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
