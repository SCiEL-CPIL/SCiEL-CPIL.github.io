import time
import random
import numpy as np

N = 256
DEADLINE = 85.0

# Precompute all 4-APs and 3-APs as bitmasks (0-indexed)
def _precompute():
    # All 4-APs: (a, a+d, a+2d, a+3d) 0-indexed
    four_aps = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                four_aps.append((a, b, c, e))
    
    # All 3-APs: (a, b, c) with a<b<c, b-a = c-b
    three_aps = []
    for a in range(N):
        for b in range(a + 1, N):
            c = 2 * b - a
            if c < N and c > b:
                three_aps.append((a, b, c))
    
    # For each element, precompute bitmask of elements that form a 4-AP with it
    # Actually, for fast feasibility: for a given set S (bitmask), check if adding x creates a 4-AP
    # Precompute for each x: list of 4-APs containing x, and for each such 4-AP, the other 3 elements
    elem_4ap_partners = [set() for _ in range(N)]
    for (a, b, c, e) in four_aps:
        elem_4ap_partners[a].update([b, c, e])
        elem_4ap_partners[b].update([a, c, e])
        elem_4ap_partners[c].update([a, b, e])
        elem_4ap_partners[e].update([a, b, c])
    
    # For 3-AP counting: for each element x, precompute bitmask of pairs (y,z) such that {x,y,z} forms a 3-AP
    # Better: for each x, precompute bitmask of elements y such that there exists z with {x,y,z} a 3-AP
    # Even better for delta computation: for each x, precompute bitmask of elements y where x,y form part of a 3-AP
    # and the third element is determined.
    
    # For fast 3-AP count: count = sum over all 3-APs of [all three in set]
    # Precompute 3-AP bitmask: for each 3-AP (a,b,c), a bitmask with bits a,b,c set
    three_aps_bitmasks = []
    for (a, b, c) in three_aps:
        mask = (1 << a) | (1 << b) | (1 << c)
        three_aps_bitmasks.append(mask)
    
    # For delta: when adding x, new 3-APs are those containing x and two other elements in S
    # Precompute for each x: list of (y, z) pairs such that {x,y,z} is a 3-AP
    x_3ap_pairs = [[] for _ in range(N)]
    for (a, b, c) in three_aps:
        x_3ap_pairs[a].append((b, c))
        x_3ap_pairs[b].append((a, c))
        x_3ap_pairs[c].append((a, b))
    
    # For delta when removing x: 3-APs lost are those containing x and two others in S
    # Same structure as above
    
    # For 4-AP feasibility check when adding x to S:
    # A 4-AP is created iff there exists a 4-AP (w,x,y,z) (in some order) where the other 3 are in S
    # Precompute for each x: list of (set of other 3 elements) for each 4-AP containing x
    x_4ap_others = [[] for _ in range(N)]
    for (a, b, c, e) in four_aps:
        x_4ap_others[a].append((b, c, e))
        x_4ap_others[b].append((a, c, e))
        x_4ap_others[c].append((a, b, e))
        x_4ap_others[e].append((a, b, c))
    
    return four_aps, three_aps, three_aps_bitmasks, x_3ap_pairs, x_4ap_others, elem_4ap_partners

(FOUR_APS, THREE_APS, THREE_APS_BITMASKS, X_3AP_PAIRS, X_4AP_OTHERS, ELEM_4AP_PARTNERS) = _precompute()

def _count_3ap_bitmask(mask):
    """Count 3-APs in a set represented as a Python int bitmask."""
    count = 0
    for bm in THREE_APS_BITMASKS:
        if (mask & bm) == bm:
            count += 1
    return count

def _check_4ap_add(mask, x):
    """Check if adding element x (0-indexed) to set mask creates a 4-AP."""
    for (y, z, w) in X_4AP_OTHERS[x]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def _delta_3ap_add(mask, x):
    """Count new 3-APs created by adding x to set mask."""
    gain = 0
    for (y, z) in X_3AP_PAIRS[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    return gain

def _delta_3ap_remove(mask, x):
    """Count 3-APs lost by removing x from set mask."""
    loss = 0
    for (y, z) in X_3AP_PAIRS[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            loss += 1
    return loss

def _backtracking_construct(start_time, deadline=40.0):
    """
    Backtracking with forward checking to build a 4-AP-free set.
    We process elements in a specific order and use constraint propagation.
    Strategy: try to build a large set by greedily including elements,
    but with backtracking when stuck.
    """
    # Order elements to process - use a heuristic: process elements that are
    # "central" first (more 4-AP constraints) to prune early
    # Actually, for maximizing 3-APs, we want to include elements that form many 3-APs
    # Let's use a different approach: iterative improvement with backtracking
    
    # Start with empty set, try to add elements in order of 3-AP potential
    # But use backtracking: if we get stuck (no more elements can be added),
    # backtrack and try a different choice
    
    best_mask = 0
    best_count = 0
    
    # Element ordering: prioritize elements with high 3-AP degree
    elem_3ap_degree = [len(X_3AP_PAIRS[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: -elem_3ap_degree[x])
    
    # Simple greedy with backtracking
    def bt(pos, mask, selected_list):
        nonlocal best_mask, best_count
        if time.time() - start_time > deadline:
            return
        
        # Count current 3-APs
        cur_count = _count_3ap_bitmask(mask)
        if cur_count > best_count:
            best_count = cur_count
            best_mask = mask
        
        if pos >= len(order):
            return
        
        x = order[pos]
        
        # Option 1: include x if feasible
        if not _check_4ap_add(mask, x):
            bt(pos + 1, mask | (1 << x), selected_list + [x])
        
        # Option 2: exclude x
        bt(pos + 1, mask, selected_list)
    
    # Full backtracking is too expensive. Use a limited-depth backtracking.
    # Instead, do a greedy pass with randomization and multiple restarts.
    
    best_mask = 0
    best_count = 0
    
    for restart in range(20):
        if time.time() - start_time > deadline - 5:
            break
        
        mask = 0
        # Random perturbation of order
        shuffled_order = order[:]
        if restart > 0:
            # Randomly shuffle a portion
            for i in range(0, N, 3):
                if i + 2 < N:
                    idx = random.randint(i, min(i + 2, N - 1))
                    shuffled_order[i], shuffled_order[idx] = shuffled_order[idx], shuffled_order[i]
        
        for x in shuffled_order:
            if time.time() - start_time > deadline - 5:
                break
            if not _check_4ap_add(mask, x):
                mask |= (1 << x)
        
        count = _count_3ap_bitmask(mask)
        if count > best_count:
            best_count = count
            best_mask = mask
    
    return best_mask, best_count

def _vectorized_local_search(mask, start_time, max_iters=50000):
    """
    Local search with swap and flip moves, using precomputed structures for O(1) feasibility and delta.
    """
    current_mask = mask
    current_score = _count_3ap_bitmask(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    T = 3.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Flip: add or remove a random element
            if random.random() < 0.5:
                # Add
                candidates = [x for x in range(N) if not (current_mask >> x) & 1]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if _check_4ap_add(current_mask, x):
                    continue
                new_mask = current_mask | (1 << x)
                delta = _delta_3ap_add(current_mask, x)
                new_score = current_score + delta
            else:
                # Remove
                selected = [x for x in range(N) if (current_mask >> x) & 1]
                if not selected:
                    continue
                x = random.choice(selected)
                new_mask = current_mask & ~(1 << x)
                delta = -_delta_3ap_remove(current_mask, x)
                new_score = current_score + delta
        else:
            # Swap: remove one, add another
            selected = [x for x in range(N) if (current_mask >> x) & 1]
            unselected = [x for x in range(N) if not (current_mask >> x) & 1]
            if not selected or not unselected:
                continue
            x = random.choice(selected)  # to remove
            y = random.choice(unselected)  # to add
            temp_mask = current_mask & ~(1 << x)
            if _check_4ap_add(temp_mask, y):
                continue
            new_mask = temp_mask | (1 << y)
            # Delta: remove x's contribution, add y's contribution (in the modified set)
            delta_remove = -_delta_3ap_remove(current_mask, x)
            delta_add = _delta_3ap_add(temp_mask, y)
            # But some 3-APs involving both x and y might be double-counted or missed
            # For simplicity, compute full delta
            new_score = _count_3ap_bitmask(new_mask)
            delta = new_score - current_score
        
        if new_score > current_score or random.random() < (2.71828 ** (delta / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
        if T < 0.05:
            T = 0.05
    
    return best_mask, best_score

def _multi_start_local_search(start_time, n_starts=5):
    """Multiple random starts with local search."""
    best_mask = 0
    best_score = 0
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Random initial set: include elements with probability 0.3, check 4-AP constraint
        mask = 0
        for x in range(N):
            if random.random() < 0.3:
                if not _check_4ap_add(mask, x):
                    mask |= (1 << x)
        
        score = _count_3ap_bitmask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
        
        # Local search
        if time.time() - start_time < DEADLINE - 5:
            lm, ls = _vectorized_local_search(mask, start_time, max_iters=20000)
            if ls > best_score:
                best_score = ls
                best_mask = lm
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Phase 1: Backtracking/greedy construction
    mask1, score1 = _backtracking_construct(start_time, deadline=30.0)
    
    # Phase 2: Multi-start local search
    mask2, score2 = _multi_start_local_search(start_time, n_starts=4)
    
    # Choose better
    if score1 >= score2:
        best_mask = mask1
        best_score = score1
    else:
        best_mask = mask2
        best_score = score2
    
    # Phase 3: Final local search polish
    if time.time() - start_time < DEADLINE - 3:
        mask3, score3 = _vectorized_local_search(best_mask, start_time, max_iters=30000)
        if score3 > best_score:
            best_mask = mask3
            best_score = score3
    
    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    
    if not result:
        result = [1]
    
    return result
