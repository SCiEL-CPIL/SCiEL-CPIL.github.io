# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _get_neighbors(p):
    """Yield 6 neighbors of point p in 3D."""
    x, y, z = p
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _energy(sequence, chain):
    """
    Compute the energy (negative number of H-H contacts) of a chain.
    Optimized for speed: single pass, direct dict lookup.
    """
    if not chain:
        return 0
    
    n = len(chain)
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            x, y, z = p
            # Check 6 neighbors manually to avoid generator overhead
            # +x
            j = pos_to_idx.get((x + 1, y, z))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
            # -x
            j = pos_to_idx.get((x - 1, y, z))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
            # +y
            j = pos_to_idx.get((x, y + 1, z))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
            # -y
            j = pos_to_idx.get((x, y - 1, z))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
            # +z
            j = pos_to_idx.get((x, y, z + 1))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
            # -z
            j = pos_to_idx.get((x, y, z - 1))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
                
    return -contacts

def _random_fold(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
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
        neighbors = list(_get_neighbors(chain[-1]))
        rng.shuffle(neighbors)
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

def _kink_move(chain, idx, rng):
    """
    Perform a kink move at index idx (1 to n-2).
    Tries to move chain[idx] to a valid neighbor of chain[idx-1] and chain[idx+1].
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if n < 3:
        return None
    if idx < 1 or idx > n - 2:
        return None
    
    p_prev = chain[idx - 1]
    p_next = chain[idx + 1]
    p_curr = chain[idx]
    
    # Find neighbors of p_prev
    neighbors_prev = list(_get_neighbors(p_prev))
    
    valid_moves = []
    for nb in neighbors_prev:
        if nb == p_curr:
            continue
        # Check if nb is a neighbor of p_next
        dx = abs(nb[0] - p_next[0])
        dy = abs(nb[1] - p_next[1])
        dz = abs(nb[2] - p_next[2])
        if dx + dy + dz == 1:
            # Check if nb is occupied by any other part of the chain
            # We need to check against all chain positions except idx
            # For efficiency, we can just check the specific points
            # But we don't have a global set here. 
            # However, kink moves are local. 
            # The only points that could conflict are those already in the chain.
            # Since we are only moving idx, we must ensure nb is not chain[i] for i != idx.
            # This requires a set of all positions. 
            # To keep this fast, we will pass an occupied set if available, 
            # or we can just check the immediate neighbors? No, long-range collisions are possible.
            # 
            # Strategy: 
            # 1. Generate candidate.
            # 2. In _sa_fold, we maintain an 'occupied' set. 
            #    We can check `nb in occupied` (excluding p_curr).
            #    But _kink_move doesn't have access to occupied set in current signature.
            #    We can modify _sa_fold to pass occupied set, or just do a linear scan? 
            #    Linear scan is O(N). N=136 is small. 
            #    Let's do a linear scan for correctness and simplicity, as kinks are frequent.
            conflict = False
            for i in range(n):
                if i == idx:
                    continue
                if chain[i] == nb:
                    conflict = True
                    break
            if not conflict:
                valid_moves.append(nb)
                
    if not valid_moves:
        return None
        
    new_pos = rng.choice(valid_moves)
    
    # Create new chain
    new_chain = list(chain)
    new_chain[idx] = new_pos
    return new_chain

def _sa_fold(sequence, dimension, time_limit, rng):
    """
    Simulated Annealing with Pivot and Kink Moves.
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
    # Increased step target to utilize faster kink moves
    total_steps_target = 30000 + n * 150
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 0.5:
            break
            
        if steps >= total_steps_target:
            break
            
        T = T0 * (Tend / T0) ** (steps / float(total_steps_target))
        
        # Mix of moves: 70% Pivot, 30% Kink
        if rng.random() < 0.7:
            pivot_idx = rng.randint(1, n - 2)
            new_chain = _pivot_move(current_chain, pivot_idx, dimension, rng)
        else:
            kink_idx = rng.randint(1, n - 2)
            new_chain = _kink_move(current_chain, kink_idx, rng)
            
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
    Multi-Start Simulated Annealing with Kink and Pivot Moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Total time budget is 300s per evaluation, but we have 26 sequences.
    # The prompt says "240 s CPU budget for all sequences together".
    # So per sequence, we have roughly 240/26 ~ 9.2 seconds.
    # We should use a conservative limit to avoid timeouts.
    total_time_limit = 8.0
    start_time = time.time()
    
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    # Increase restarts to 5 to leverage the faster local moves
    num_restarts = 5
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
