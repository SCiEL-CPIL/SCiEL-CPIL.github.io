"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import math

DEADLINE = time.time() + 290  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict bitsets:
# conflict[p][a] is a bitmask of all points b such that (p, a, b) forms an isosceles triangle
conflict = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for p in range(NUM_PTS):
    for a in range(NUM_PTS):
        if p == a:
            continue
        mask = 0
        for b in range(NUM_PTS):
            if b == p or b == a:
                continue
            d_pa = dist[p][a]
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << b)
        conflict[p][a] = mask

# Precompute: for each point p and each pair (a,b), does adding p create isosceles?
# We use: p can be added to set S iff for all a in S, conflict[p][a] & S == 0
# (i.e., no b in S such that (p,a,b) is isosceles)

def can_add(p, chosen_mask):
    """Check if point p can be added given chosen points represented as bitmask."""
    cm = chosen_mask
    while cm:
        lb = cm & (-cm)
        a = lb.bit_length() - 1
        if conflict[p][a] & chosen_mask:
            return False
        cm ^= lb
    return True

def is_valid_set(mask):
    """Check if the set represented by mask is a valid no-isosceles set."""
    pts = []
    cm = mask
    while cm:
        lb = cm & (-cm)
        p = lb.bit_length() - 1
        pts.append(p)
        cm ^= lb
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            a, b = pts[i], pts[j]
            if conflict[a][b] & mask:
                return False
    return True

def count_blocked(mask):
    """Count unselected points that are 'blocked' (form isosceles with two selected points).
    A point p is blocked if there exist a, b in selected such that (p,a,b) is isosceles."""
    blocked = 0
    cm = mask
    selected = []
    while cm:
        lb = cm & (-cm)
        p = lb.bit_length() - 1
        selected.append(p)
        cm ^= lb
    
    for p in range(NUM_PTS):
        if (mask >> p) & 1:
            continue
        # Check if p is blocked: exists a,b in selected with (p,a,b) isosceles
        is_blocked = False
        for a in selected:
            if conflict[p][a] & mask:
                is_blocked = True
                break
        if is_blocked:
            blocked += 1
    return blocked

def greedy_construct(order, chosen_mask=0):
    """Greedy construction following given order of points."""
    chosen_mask = 0
    for p in order:
        if not (chosen_mask >> p) & 1:
            if can_add(p, chosen_mask):
                chosen_mask |= (1 << p)
    return chosen_mask

def complete_to_maximal(chosen_mask, rng):
    """Add points until the set is maximal (no more can be added)."""
    candidates = [p for p in range(NUM_PTS) if not (chosen_mask >> p) & 1]
    rng.shuffle(candidates)
    for p in candidates:
        if can_add(p, chosen_mask):
            chosen_mask |= (1 << p)
    return chosen_mask

def randomized_greedy(seed):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    mask = greedy_construct(order)
    # Ensure maximality
    mask = complete_to_maximal(mask, rng)
    return mask

def simulated_annealing(initial_mask, max_iters=5000):
    """Simulated annealing to minimize |selected| while maintaining maximality.
    
    State: bitmask of selected points.
    Moves: 
      1. Remove a random selected point (if set remains valid)
      2. Add a random unselected point (if valid)
      3. Swap: remove one, add another
    Energy: |selected| + penalty * (unblocked_count)
    We want to minimize energy.
    """
    rng = random.Random(12345)
    current_mask = initial_mask
    current_size = bin(current_mask).count('1')
    current_unblocked = NUM_PTS - current_size - count_blocked(current_mask)
    # Energy = size + 0.5 * unblocked (we want small size AND few unblocked)
    current_energy = current_size + 0.5 * current_unblocked
    
    best_mask = current_mask
    best_energy = current_energy
    
    T = 5.0  # Initial temperature
    cooling = 0.999  # Cooling rate
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        new_mask = current_mask
        
        if move_type < 0.4:
            # Try removing a random selected point
            selected_pts = []
            cm = current_mask
            while cm:
                lb = cm & (-cm)
                p = lb.bit_length() - 1
                selected_pts.append(p)
                cm ^= lb
            if not selected_pts:
                continue
            p = rng.choice(selected_pts)
            new_mask = current_mask & ~(1 << p)
            # Check validity
            if not is_valid_set(new_mask):
                continue
            # After removal, the set may no longer be maximal. 
            # We allow non-maximal states during SA but penalize unblocked points.
            
        elif move_type < 0.7:
            # Try adding a random unselected point
            unselected = [p for p in range(NUM_PTS) if not (current_mask >> p) & 1]
            if not unselected:
                continue
            p = rng.choice(unselected)
            if not can_add(p, current_mask):
                continue
            new_mask = current_mask | (1 << p)
            
        else:
            # Swap: remove one selected, add one unselected
            selected_pts = []
            cm = current_mask
            while cm:
                lb = cm & (-cm)
                p = lb.bit_length() - 1
                selected_pts.append(p)
                cm ^= lb
            unselected = [p for p in range(NUM_PTS) if not (current_mask >> p) & 1]
            if not selected_pts or not unselected:
                continue
            p_remove = rng.choice(selected_pts)
            p_add = rng.choice(unselected)
            new_mask = (current_mask & ~(1 << p_remove)) | (1 << p_add)
            # Check validity of the new set
            if not is_valid_set(new_mask):
                continue
        
        new_size = bin(new_mask).count('1')
        new_unblocked = NUM_PTS - new_size - count_blocked(new_mask)
        new_energy = new_size + 0.5 * new_unblocked
        
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_mask = new_mask
            current_size = new_size
            current_unblocked = new_unblocked
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_mask = new_mask
                best_energy = new_energy
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_mask

def local_search(chosen_mask, max_iterations=50):
    """Local search: try removing a point and refilling to maximality, keep if smaller."""
    best_mask = chosen_mask
    best_len = bin(best_mask).count('1')
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        selected_pts = []
        cm = best_mask
        while cm:
            lb = cm & (-cm)
            p = lb.bit_length() - 1
            selected_pts.append(p)
            cm ^= lb
        
        rng = random.Random(iteration * 10000 + 42)
        rng.shuffle(selected_pts)
        
        for removed in selected_pts:
            if time.time() > DEADLINE:
                break
            
            new_mask = best_mask & ~(1 << removed)
            
            # Complete to maximality with a fresh random order
            new_mask = complete_to_maximal(new_mask, rng)
            
            new_len = bin(new_mask).count('1')
            if new_len < best_len:
                best_mask = new_mask
                best_len = new_len
                improved = True
                break
        
        if not improved:
            break
    
    return best_mask

def shrink_search(chosen_mask):
    """Try to shrink the set by removing points one at a time and re-completing."""
    best_mask = chosen_mask
    best_len = bin(best_mask).count('1')
    
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        selected_pts = []
        cm = best_mask
        while cm:
            lb = cm & (-cm)
            p = lb.bit_length() - 1
            selected_pts.append(p)
            cm ^= lb
        
        for removed in selected_pts:
            if time.time() > DEADLINE:
                break
            
            new_mask = best_mask & ~(1 << removed)
            
            # Try multiple random completions to find the smallest maximal superset
            best_completion = 0
            best_completion_len = float('inf')
            
            for attempt in range(5):
                rng = random.Random(attempt * 1000 + removed + 12345)
                cm2 = complete_to_maximal(new_mask, rng)
                l2 = bin(cm2).count('1')
                if l2 < best_completion_len:
                    best_completion = cm2
                    best_completion_len = l2
            
            if best_completion_len < best_len:
                best_mask = best_completion
                best_len = best_completion_len
                changed = True
                break
    
    return best_mask

def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best_mask = 0
    best_len = float('inf')
    
    # Baseline: deterministic row-major greedy
    baseline_order = list(range(NUM_PTS))
    baseline_mask = greedy_construct(baseline_order)
    baseline_mask = complete_to_maximal(baseline_mask, random.Random(0))
    best_mask = baseline_mask
    best_len = bin(best_mask).count('1')
    
    # Randomized restarts with local search
    seed = 42
    while time.time() < DEADLINE - 30:
        mask = randomized_greedy(seed)
        cur_len = bin(mask).count('1')
        
        if cur_len < best_len:
            best_mask = mask
            best_len = cur_len
        
        # Local search on this solution
        if cur_len <= best_len:
            improved_mask = local_search(mask, max_iterations=30)
            imp_len = bin(improved_mask).count('1')
            if imp_len < best_len:
                best_mask = improved_mask
                best_len = imp_len
        
        seed += 1
    
    # Simulated annealing on the best solution found
    if time.time() < DEADLINE - 20:
        sa_mask = simulated_annealing(best_mask, max_iters=3000)
        sa_len = bin(sa_mask).count('1')
        if sa_len < best_len:
            best_mask = sa_mask
            best_len = sa_len
    
    # Final polish: shrink search on the best solution found
    if time.time() < DEADLINE - 5:
        polished_mask = shrink_search(best_mask)
        pol_len = bin(polished_mask).count('1')
        if pol_len < best_len:
            best_mask = polished_mask
            best_len = pol_len
    
    # Convert mask back to list of points
    result = []
    cm = best_mask
    while cm:
        lb = cm & (-cm)
        p = lb.bit_length() - 1
        result.append(POINTS[p])
        cm ^= lb
    
    return result
# EVOLVE-BLOCK-END
