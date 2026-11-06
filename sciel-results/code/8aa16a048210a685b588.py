import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}

# Distance matrix
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute all isosceles triples: for each pair (a,b), find all c such that (a,b,c) is isosceles
# Store as: for each point c, the set of pairs (a,b) that form isosceles with c
# This helps in checking if c is blocked by a selected set

def _is_no_isosceles_fast(indices):
    """Check if set of point indices forms no-isosceles set using vectorized ops."""
    n = len(indices)
    if n < 3:
        return True
    idx = np.array(indices, dtype=np.int32)
    # Compute all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = _D2[idx[i], idx[j]]
            for k in range(j + 1, n):
                d_ik = _D2[idx[i], idx[k]]
                d_jk = _D2[idx[j], idx[k]]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True

def _blocked_points(chosen_indices):
    """Return boolean array: which points are blocked by chosen_indices."""
    n = len(chosen_indices)
    blocked = np.zeros(N_TOTAL, dtype=bool)
    if n < 2:
        return blocked
    
    ch = np.array(chosen_indices, dtype=np.int32)
    
    # For each point p not in chosen, check if it forms isosceles with any pair in chosen
    for p in range(N_TOTAL):
        if p in chosen_indices:
            blocked[p] = True
            continue
        
        # Check if p forms isosceles with any pair (a,b) in chosen
        # d(p,a) == d(p,b) or d(p,a) == d(a,b) or d(p,b) == d(a,b)
        dp = _D2[p, ch]  # distances from p to each chosen point
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                da = dp[i]
                db = dp[j]
                dab = _D2[ch[i], ch[j]]
                if da == db or da == dab or db == dab:
                    found = True
                    break
            if found:
                break
        blocked[p] = found
    
    return blocked

def _can_add_point(idx_p, chosen_indices):
    """Check if point idx_p can be added to chosen without creating isosceles triangle."""
    if len(chosen_indices) < 2:
        return True
    ch = np.array(chosen_indices, dtype=np.int32)
    dp = _D2[idx_p, ch]
    for i in range(len(ch)):
        for j in range(i + 1, len(ch)):
            da = dp[i]
            db = dp[j]
            dab = _D2[ch[i], ch[j]]
            if da == db or da == dab or db == dab:
                return False
    return True

def _is_maximal_fast(chosen_indices):
    """Check if chosen set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for i in range(N_TOTAL):
        if i in chosen_set:
            continue
        if _can_add_point(i, chosen_indices):
            return False
    return True

def _reduce_to_minimal(chosen_indices, time_budget=2.0):
    """Iteratively remove points if remaining set is still maximal."""
    start = time.time()
    S = list(chosen_indices)
    changed = True
    while changed:
        changed = False
        if time.time() - start > time_budget:
            break
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i + 1:]
            if _is_maximal_fast(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_hitting_set(deadline_margin=10):
    """
    Greedy hitting set approach:
    - Start with empty set
    - At each step, add the point that blocks the most currently-unblocked points
    - While maintaining no-isosceles property
    """
    chosen = []
    chosen_set = set()
    
    while True:
        # Find unblocked points (not in chosen, not blocked by chosen)
        blocked = _blocked_points(chosen)
        unblocked = [i for i in range(N_TOTAL) if i not in chosen_set and not blocked[i]]
        
        if not unblocked:
            break
        
        # Check if already maximal
        if _is_maximal_fast(chosen):
            break
        
        # Find all addable points
        addable = []
        for i in unblocked:
            if _can_add_point(i, chosen):
                addable.append(i)
        
        if not addable:
            break
        
        # Score each addable point: how many unblocked points would it block?
        best_p = None
        best_gain = -1
        
        for p in addable:
            # Simulate adding p
            test_chosen = chosen + [p]
            test_blocked = _blocked_points(test_chosen)
            # Count how many previously-unblocked points are now blocked
            gain = 0
            for q in unblocked:
                if q != p and test_blocked[q]:
                    gain += 1
            # Also count p itself
            gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        if time.time() > DEADLINE - deadline_margin:
            return None
    
    return chosen

def _local_search_improve(S, time_budget=5.0):
    """Try to improve set S via local search (remove/add points)."""
    start = time.time()
    S_list = list(S)
    best_size = len(S_list)
    best_set = S_list[:]
    
    while time.time() - start < time_budget:
        improved = False
        
        # Try removing one point
        for i in range(len(S_list)):
            S_test = S_list[:i] + S_list[i + 1:]
            if len(S_test) >= 2 and _is_maximal_fast(S_test):
                S_list = S_test
                best_size = len(S_list)
                best_set = S_list[:]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for i in range(len(S_list)):
            S_without = S_list[:i] + S_list[i + 1:]
            if len(S_without) < 2:
                continue
            
            for p_add in range(N_TOTAL):
                if p_add in S_without:
                    continue
                if not _can_add_point(p_add, S_without):
                    continue
                
                S_new = S_without + [p_add]
                if _is_maximal_fast(S_new) and len(S_new) < best_size:
                    S_list = S_new
                    best_size = len(S_new)
                    best_set = S_list[:]
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_set

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy hitting set from scratch
    result = _greedy_hitting_set(deadline_margin=15)
    if result is not None:
        reduced = _reduce_to_minimal(result, time_budget=3.0)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 2: Try random cores and build up
    core_sizes = [2, 3, 4, 5, 6, 7, 8]
    
    for core_size in core_sizes:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        n_attempts = 100
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            if not _is_no_isosceles_fast(core_indices):
                continue
            
            # Build maximal set from core using greedy
            chosen = list(core_indices)
            chosen_set = set(chosen)
            
            while True:
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                blocked = _blocked_points(chosen)
                unblocked = [i for i in range(N_TOTAL) if i not in chosen_set and not blocked[i]]
                
                if not unblocked:
                    break
                
                if _is_maximal_fast(chosen):
                    break
                
                addable = [i for i in unblocked if _can_add_point(i, chosen)]
                if not addable:
                    break
                
                # Pick point that blocks most
                best_p = None
                best_gain = -1
                for p in addable:
                    test_chosen = chosen + [p]
                    test_blocked = _blocked_points(test_chosen)
                    gain = sum(1 for q in unblocked if q != p and test_blocked[q]) + 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                
                if best_p is None:
                    break
                
                chosen.append(best_p)
                chosen_set.add(best_p)
            
            if _is_maximal_fast(chosen):
                reduced = _reduce_to_minimal(chosen, time_budget=2.0)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 8:
                break
    
    # Strategy 3: Random greedy + reduction
    for attempt in range(300):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        order = list(range(N_TOTAL))
        rng.shuffle(order)
        
        chosen = []
        for i in order:
            if _can_add_point(i, chosen):
                chosen.append(i)
        
        if _is_maximal_fast(chosen):
            reduced = _reduce_to_minimal(chosen, time_budget=1.5)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
        
        if best_size <= 8:
            break
    
    # Strategy 4: Local search on best set
    if best_set is not None and best_size > 8:
        improved_set = _local_search_improve(best_set, time_budget=10.0)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    if best_set is None:
        # Fallback
        chosen = []
        for i in range(N_TOTAL):
            if _can_add_point(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen, time_budget=5.0)
    
    return [_ALL_POINTS[i] for i in best_set]
