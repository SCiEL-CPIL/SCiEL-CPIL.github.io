# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """
    Compute the energy (negative number of H-H contacts) of a chain.
    """
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _random_fold(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    If it gets stuck, it returns None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        random.shuffle(neighbors)
        valid = False
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied.add(nb)
                valid = True
                break
        if not valid:
            return None
    return chain

def _pivot_move(chain, pivot_idx, dimension, rng):
    """
    Perform a pivot move at pivot_idx.
    Returns the new chain if valid, else None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    if pivot_idx < 1 or pivot_idx > n - 2:
        return None
        
    p_pivot = chain[pivot_idx]
    
    rot_axes = [0, 1, 2]
    r_axis = rng.choice(rot_axes)
    angle = rng.choice([1, -1])
    
    new_tail = []
    valid = True
    occupied_tail = set()
    head_set = set(chain[:pivot_idx])
    
    for i in range(pivot_idx + 1, n):
        orig = chain[i]
        dx = orig[0] - p_pivot[0]
        dy = orig[1] - p_pivot[1]
        dz = orig[2] - p_pivot[2]
        
        if r_axis == 0:
            if angle == 1:
                nx, ny, nz = dx, -dz, dy
            else:
                nx, ny, nz = dx, dz, -dy
        elif r_axis == 1:
            if angle == 1:
                nx, ny, nz = dz, dy, -dx
            else:
                nx, ny, nz = -dz, dy, dx
        else:
            if angle == 1:
                nx, ny, nz = -dy, dx, dz
            else:
                nx, ny, nz = dy, -dx, dz
                
        new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
        
        if new_point in head_set or new_point in occupied_tail:
            valid = False
            break
            
        occupied_tail.add(new_point)
        new_tail.append(new_point)
        
    if not valid:
        return None
        
    return chain[:pivot_idx + 1] + new_tail

def _sa_fold(sequence, dimension, time_limit, rng):
    """
    Simulated Annealing with Pivot Moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    current_chain = None
    for _ in range(100):
        current_chain = _random_fold(sequence, dimension, rng)
        if current_chain:
            break
            
    if current_chain is None:
        current_chain = [(i, 0, 0) for i in range(n)]
        
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T0 = 5.0
    Tend = 0.01
    start_time = time.time()
    steps = 0
    total_steps_target = 20000 + n * 100
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 1.0:
            break
            
        if steps >= total_steps_target:
            break
            
        T = T0 * (Tend / T0) ** (steps / float(total_steps_target))
        
        pivot_idx = rng.randint(1, n - 2)
        new_chain = _pivot_move(current_chain, pivot_idx, dimension, rng)
        
        if new_chain is None:
            steps += 1
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        else:
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
            
        steps += 1
        
    return best_chain

def fold(sequence, dimension):
    """
    Multi-Start Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    total_time_limit = 8.5
    start_time = time.time()
    
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    num_restarts = 3
    time_per_run = total_time_limit / num_restarts
    
    for r in range(num_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
            
        rng = random.Random(12345 + len(sequence) * 100 + r)
        
        t_limit = time_per_run if r < num_restarts - 1 else remaining
        chain = _sa_fold(sequence, dimension, t_limit, rng)
        
        if chain:
            energy = _energy(sequence, chain)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_chain_overall = chain
                
    if best_chain_overall is None:
        best_chain_overall = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain_overall]

# EVOLVE-BLOCK-END
