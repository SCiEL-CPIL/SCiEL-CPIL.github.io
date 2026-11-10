import random
import time
import math

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _rotate(pos, center, angle):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    if angle == 90:
        new_dx, new_dy = -dy, dx
    elif angle == 180:
        new_dx, new_dy = -dx, -dy
    elif angle == 270:
        new_dx, new_dy = dy, -dx
    else:
        return pos
    return (center[0] + new_dx, center[1] + new_dy)

def _initial_growth(sequence, rng):
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours(last)
        random.shuffle(neighbors)
        placed = False
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied.add(nb)
                placed = True
                break
        if not placed:
            return None
    return chain

def _calc_energy_fast(sequence, chain):
    n = len(chain)
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            j = pos_to_idx.get((x+1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                energy += 1
            j = pos_to_idx.get((x-1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                energy += 1
            j = pos_to_idx.get((x, y+1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                energy += 1
            j = pos_to_idx.get((x, y-1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                energy += 1
    return -energy

def _sa_fold_fast(sequence, chain, deadline):
    n = len(chain)
    if n < 3:
        return chain, _calc_energy_fast(sequence, chain)
    
    h_indices = {i for i, c in enumerate(sequence) if c == 'H'}
    
    current_chain = list(chain)
    current_energy = _calc_energy_fast(sequence, current_chain)
    
    T = 1.0
    T_min = 0.01
    steps = 50000
    cooling = (T_min / T) ** (1.0 / steps)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        k = random.randint(1, n - 2)
        angle = random.choice([90, 180, 270])
        pivot_pos = current_chain[k]
        
        head_positions = set(current_chain[:k+1])
        h_head_positions = set()
        for i in range(k + 1):
            if i in h_indices:
                h_head_positions.add(current_chain[i])
                
        new_tail_positions = []
        valid = True
        for i in range(k + 1, n):
            old_pos = current_chain[i]
            new_pos = _rotate(old_pos, pivot_pos, angle)
            new_tail_positions.append(new_pos)
            if new_pos in head_positions:
                valid = False
                break
        
        if not valid:
            continue
            
        delta_e = 0
        
        # Remove old contacts
        for i in range(k + 1, n):
            if i in h_indices:
                pos = current_chain[i]
                # Unroll neighbors for speed
                x, y = pos
                if (x+1, y) in h_head_positions: delta_e -= 1
                if (x-1, y) in h_head_positions: delta_e -= 1
                if (x, y+1) in h_head_positions: delta_e -= 1
                if (x, y-1) in h_head_positions: delta_e -= 1
                
        # Add new contacts
        for idx, i in enumerate(range(k + 1, n)):
            if i in h_indices:
                pos = new_tail_positions[idx]
                x, y = pos
                if (x+1, y) in h_head_positions: delta_e += 1
                if (x-1, y) in h_head_positions: delta_e += 1
                if (x, y+1) in h_head_positions: delta_e += 1
                if (x, y-1) in h_head_positions: delta_e += 1
                
        new_energy = current_energy + delta_e
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_chain = current_chain[:k+1] + new_tail_positions
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling
        
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    # Budget is 120s total for 10 sequences -> 12s per sequence.
    # Use 10s to be safe.
    deadline = start_time + 10.0
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    best_chain = None
    best_energy = 0
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        init_chain = _initial_growth(sequence, rng)
        if init_chain is None:
            continue
            
        sa_chain, sa_energy = _sa_fold_fast(sequence, init_chain, deadline)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
            
        if time.time() > deadline:
            break
            
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
        
    return best_chain
