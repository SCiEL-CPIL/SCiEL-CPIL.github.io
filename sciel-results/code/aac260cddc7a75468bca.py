"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time

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


def greedy_construct(order, chosen_mask=0, chosen_list=None):
    """Greedy construction following given order of points."""
    if chosen_list is None:
        chosen_list = []
    for p in order:
        if not (chosen_mask >> p) & 1:
            if can_add(p, chosen_mask):
                chosen_list.append(p)
                chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def complete_to_maximal(chosen_list, chosen_mask, rng):
    """Add points until the set is maximal (no more can be added)."""
    candidates = [p for p in range(NUM_PTS) if not (chosen_mask >> p) & 1]
    rng.shuffle(candidates)
    for p in candidates:
        if can_add(p, chosen_mask):
            chosen_list.append(p)
            chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def randomized_greedy(seed):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    chosen, mask = greedy_construct(order)
    # Ensure maximality
    chosen, mask = complete_to_maximal(chosen, mask, rng)
    return chosen, mask


def local_search(chosen_list, chosen_mask, max_iterations=100):
    """Local search: try removing a point and refilling to maximality."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point in the current best set
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Complete to maximality with a fresh random order
            rng = random.Random(iteration * 10000 + i * 7 + 42)
            new_list, new_mask = complete_to_maximal(new_list, new_mask, rng)
            
            if len(new_list) < best_len:
                best = new_list
                best_mask = new_mask
                best_len = len(best)
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best, best_mask


def shrink_search(chosen_list, chosen_mask):
    """Try to shrink the set by removing points one at a time and re-completing."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try multiple random completions to find the smallest maximal superset
            best_completion = None
            best_completion_mask = 0
            best_completion_len = float('inf')
            
            for attempt in range(5):
                rng = random.Random(attempt * 1000 + i + 12345)
                cl, cm = complete_to_maximal(list(new_list), new_mask, rng)
                if len(cl) < best_completion_len:
                    best_completion = cl
                    best_completion_mask = cm
                    best_completion_len = len(cl)
            
            if best_completion is not None and best_completion_len < best_len:
                best = best_completion
                best_mask = best_completion_mask
                best_len = best_completion_len
                changed = True
                break
    
    return best, best_mask


def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best = None
    best_mask = 0
    best_len = float('inf')
    
    # Baseline: deterministic row-major greedy
    baseline_order = list(range(NUM_PTS))
    baseline, baseline_mask = greedy_construct(baseline_order)
    baseline, baseline_mask = complete_to_maximal(baseline, baseline_mask, random.Random(0))
    best = baseline
    best_mask = baseline_mask
    best_len = len(best)
    
    # Randomized restarts with local search
    seed = 42
    while time.time() < DEADLINE - 5:
        chosen, mask = randomized_greedy(seed)
        if len(chosen) < best_len:
            best = chosen
            best_mask = mask
            best_len = len(best)
        
        # Local search on this solution
        if len(chosen) <= best_len:
            improved, improved_mask = local_search(chosen, mask, max_iterations=30)
            if len(improved) < best_len:
                best = improved
                best_mask = improved_mask
                best_len = len(best)
        
        seed += 1
    
    # Final polish: shrink search on the best solution found
    if time.time() < DEADLINE - 2:
        polished, polished_mask = shrink_search(best, best_mask)
        if len(polished) < best_len:
            best = polished
            best_mask = polished_mask
            best_len = len(best)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
