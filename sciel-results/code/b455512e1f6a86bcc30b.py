# EVOLVE-BLOCK-START
import random
import math
import time

def fold(sequence, dimension):
    """
    Fold the HP sequence into a low-energy conformation.
    Uses Monte Carlo search with Kink and Pivot moves.
    """
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    # Time management
    global _total_start_time
    if '_total_start_time' not in globals():
        _total_start_time = time.time()
    
    remaining_time = 110.0 - (time.time() - _total_start_time)
    if remaining_time <= 0:
        return _straight_line(n)
        
    time_budget = min(8.0, remaining_time * 0.9)
    
    # 1. Initialization: Random Walk
    chain = _init_random_walk(n)
    if chain is None:
        return _straight_line(n)
        
    occupied = set(chain)
    current_energy = _calc_energy(sequence, chain)
    best_chain = [list(p) for p in chain]
    best_energy = current_energy
    
    # 2. Monte Carlo Search
    T_start = 2.0
    T_end = 0.05
    T = T_start
    cooling_rate = 0.9999
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_budget:
        # Choose move
        r = random.random()
        new_chain = None
        
        if r < 0.7:
            # Kink Move
            i = random.randint(1, n - 2)
            new_chain = _try_kink(chain, occupied, i)
        elif r < 0.9:
            # Pivot Move
            k = random.randint(1, n - 1)
            new_chain = _try_pivot(chain, occupied, k)
        else:
            # End Move
            new_chain = _try_end_move(chain, occupied)
            
        if new_chain is not None:
            new_energy = _calc_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                chain = new_chain
                occupied = set(chain)
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = [list(p) for p in chain]
                    
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_chain

def _straight_line(n):
    return [[i, 0] for i in range(n)]

def _init_random_walk(n):
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for _ in range(1, n):
        last = chain[-1]
        valid = []
        for d in dirs:
            nxt = (last[0] + d[0], last[1] + d[1])
            if nxt not in occupied:
                valid.append(nxt)
        if not valid:
            return None
        nxt = random.choice(valid)
        chain.append(nxt)
        occupied.add(nxt)
        
    return chain

def _calc_energy(sequence, chain):
    n = len(chain)
    point_to_idx = {p: i for i, p in enumerate(chain)}
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == 'H':
            h_positions.add(chain[i])
            
    contacts = 0
    for i, c in enumerate(sequence):
        if c == 'H':
            p = chain[i]
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (p[0]+dx, p[1]+dy)
                if q in point_to_idx:
                    j = point_to_idx[q]
                    if abs(i - j) > 1:
                        contacts += 1
    return -contacts // 2

def _try_kink(chain, occupied, i):
    n = len(chain)
    if i < 1 or i > n - 2:
        return None
        
    p_prev = chain[i-1]
    p_curr = chain[i]
    p_next = chain[i+1]
    
    ux = p_curr[0] - p_prev[0]
    uy = p_curr[1] - p_prev[1]
    vx = p_next[0] - p_curr[0]
    vy = p_next[1] - p_curr[1]
    
    if ux * vx + uy * vy != 0:
        return None
        
    new_p = (p_prev[0] + vx, p_prev[1] + vy)
    
    if new_p in occupied:
        return None
        
    new_chain = chain.copy()
    new_chain[i] = new_p
    return new_chain

def _try_pivot(chain, occupied, k):
    """
    Pivot the segment [0..k] around chain[k].
    """
    n = len(chain)
    if k < 1 or k >= n:
        return None
        
    pivot = chain[k]
    # Rotate segment [0..k] by 90, 180, 270
    # We try one random rotation
    angle = random.choice([1, 2, 3]) # 90, 180, 270
    
    # Rotation matrix for 90 deg: (x, y) -> (y, -x) relative to pivot?
    # Let's use vector rotation.
    
    def rotate_point(p, pivot, angle):
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        if angle == 1: # 90
            nx, ny = -y, x
        elif angle == 2: # 180
            nx, ny = -x, -y
        else: # 270
            nx, ny = y, -x
        return (pivot[0] + nx, pivot[1] + ny)
        
    new_segment = [rotate_point(p, pivot, angle) for p in chain[:k+1]]
    
    # Check collisions
    # The pivot point itself doesn't move, so it's fine.
    # We need to check if any new point in new_segment[0:k] is in occupied (excluding the points being moved).
    
    # Points being moved: chain[0..k-1]
    # Points staying: chain[k..n-1]
    
    # We need to check if any p in new_segment[0:k] is in occupied AND not in the set of points being moved.
    # Actually, occupied contains ALL points.
    # We must check if new_p is in occupied.
    # But the points being moved are still in occupied until we update it.
    # So we must check if new_p is in occupied AND new_p is not in the set of OLD positions of the moved segment.
    
    old_positions = set(chain[:k]) # Exclude pivot k as it doesn't move? 
    # Wait, chain[k] is the pivot. It stays.
    # So we move chain[0..k-1].
    # new_segment[0..k-1] are the new positions.
    # new_segment[k] should be pivot.
    
    valid = True
    for i in range(k):
        np = new_segment[i]
        if np in occupied and np not in old_positions:
            valid = False
            break
        if i > 0:
            # Check connectivity within the new segment
            prev_np = new_segment[i-1]
            if abs(np[0]-prev_np[0]) + abs(np[1]-prev_np[1]) != 1:
                valid = False
                break
    if not valid:
        return None
        
    # Check connectivity between new_segment[k-1] and chain[k]
    if k > 0:
        last_moved = new_segment[k-1]
        if abs(last_moved[0]-pivot[0]) + abs(last_moved[1]-pivot[1]) != 1:
            return None
            
    new_chain = new_segment + chain[k+1:]
    return new_chain

def _try_end_move(chain, occupied):
    n = len(chain)
    last = chain[-1]
    prev = chain[-2]
    
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    valid = []
    for d in dirs:
        nxt = (last[0] + d[0], last[1] + d[1])
        if nxt not in occupied and nxt != prev:
            valid.append(nxt)
            
    if not valid:
        return None
        
    new_p = random.choice(valid)
    new_chain = chain.copy()
    new_chain[-1] = new_p
    return new_chain
# EVOLVE-BLOCK-END
