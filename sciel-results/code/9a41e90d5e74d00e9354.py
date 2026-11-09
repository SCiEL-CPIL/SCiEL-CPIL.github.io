import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

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
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _rotate_90(vec, axis):
    vx, vy, vz = vec
    ax, ay, az = axis
    cx = ay * vz - az * vy
    cy = az * vx - ax * vz
    cz = ax * vy - ay * vx
    return (cx, cy, cz)

def _pivot_move(sequence, chain, i, rng, occupied):
    """
    Perform a pivot move at index i.
    Returns (new_chain, delta_e, valid)
    """
    N = len(chain)
    p_prev = chain[i-1]
    p_curr = chain[i]
    axis = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
    
    # Try both +90 and -90 rotations
    sign = 1 if rng.random() < 0.5 else -1
    
    moving_old = chain[i:]
    fixed_set = set(chain[:i])
    
    new_positions = []
    valid = True
    
    for j in range(i, N):
        p_old = chain[j]
        v = (p_old[0] - p_prev[0], p_old[1] - p_prev[1], p_old[2] - p_prev[2])
        rx, ry, rz = _rotate_90(v, axis)
        if sign == -1:
            rx, ry, rz = -rx, -ry, -rz
        p_new = (p_prev[0] + rx, p_prev[1] + ry, p_prev[2] + rz)
        
        if p_new in fixed_set:
            valid = False
            break
        
        new_positions.append(p_new)
    
    if not valid:
        return None, 0, False
    
    # Calculate Delta Energy
    # Only cross contacts change
    old_cross = 0
    for u in range(i):
        if sequence[u] != 'H': continue
        pu = chain[u]
        for v in range(i, N):
            if sequence[v] != 'H': continue
            pv = chain[v]
            if abs(pu[0]-pv[0]) + abs(pu[1]-pv[1]) + abs(pu[2]-pv[2]) == 1:
                old_cross += 1
                
    new_cross = 0
    for u in range(i):
        if sequence[u] != 'H': continue
        pu = chain[u]
        for k, pv_new in enumerate(new_positions):
            v_idx = i + k
            if sequence[v_idx] != 'H': continue
            if abs(pu[0]-pv_new[0]) + abs(pu[1]-pv_new[1]) + abs(pu[2]-pv_new[2]) == 1:
                new_cross += 1
                
    delta_e = -(new_cross - old_cross)
    
    new_chain = chain[:]
    new_chain[i:] = new_positions
    
    return new_chain, delta_e, True

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0: return []
    
    rng = random.Random(12345 + N)
    
    start_time = time.time()
    # Budget is 240s for all sequences. Assume ~9s per sequence.
    # Leave safety margin.
    deadline = start_time + 8.5
    
    # 1. Initialize with multiple greedy growths to get a good starting point
    best_chain = None
    best_e = 0
    
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_e:
            best_e = e
            best_chain = chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
        best_e = _energy(sequence, best_chain)
    
    current_chain = best_chain
    current_e = best_e
    
    # Simulated Annealing Parameters
    # T0 based on chain length
    T = max(2.0, N * 0.1)
    T_min = 0.01
    cooling_rate = 0.999
    
    occupied = set(current_chain)
    
    while time.time() < deadline:
        # Choose pivot index i (1 to N-2)
        i = rng.randint(1, N - 2)
        
        new_chain, delta_e, valid = _pivot_move(sequence, current_chain, i, rng, occupied)
        
        if not valid:
            continue
        
        # Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Update occupied set
            # Remove old moving part
            for p in current_chain[i:]:
                occupied.remove(p)
            # Add new moving part
            for p in new_chain[i:]:
                occupied.add(p)
                
            current_chain = new_chain
            current_e += delta_e
            
            if current_e < best_e:
                best_e = current_e
                # No need to copy best_chain if we just updated current_chain and it's better
                # But we need to keep a separate reference if we revert later
                # For simplicity, we only track best_e and best_chain
                # If we reject later, best_chain is still valid
                # Wait, if we accept a move that is worse than best, we don't update best.
                # If we accept a move that is better, we update best.
                # But current_chain is now the new one.
                # We need to store best_chain explicitly if we want to return it.
                # Let's just copy when we find a new best.
                best_chain = list(current_chain)
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Rejected move, no change to current_chain or occupied
            pass
            
    return [list(p) for p in best_chain]
