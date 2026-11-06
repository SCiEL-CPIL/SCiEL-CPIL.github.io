import time
import random

N = 256

def _construct_maximal(order, N_val=N):
    """
    Construct a maximal Sidon set by iterating 'order' and adding elements if possible.
    Returns (list, set_of_sums).
    """
    selected = []
    sums = set()
    for x in order:
        if 2 * x in sums:
            continue
        
        conflict = False
        for s in selected:
            if (x + s) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        selected.append(x)
        sums.add(2 * x)
        for s in selected[:-1]:
            sums.add(x + s)
            
    return selected, sums

def _is_maximal(set_list, sums, N_val=N):
    """
    Check if set_list is inclusion-maximal Sidon in [1, N_val].
    """
    s_set = set(set_list)
    for x in range(1, N_val + 1):
        if x in s_set:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _reduce_maximal(set_list, sums, deadline, rng):
    """
    Try to remove elements while preserving maximality.
    Returns (reduced_list, reduced_sums).
    """
    current = list(set_list)
    current_sums = set(sums)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                return current, current_sums
            
            # Try removing current[idx]
            # We don't need to check Sidon property, only maximality.
            # Removing an element from a Sidon set results in a Sidon set.
            
            new_current = current[:idx] + current[idx+1:]
            
            # Recompute sums for new_current
            # O(k^2) where k is len(new_current)
            new_sums = set()
            for i in range(len(new_current)):
                ai = new_current[i]
                for j in range(i, len(new_current)):
                    new_sums.add(ai + new_current[j])
            
            if _is_maximal(new_current, new_sums):
                current = new_current
                current_sums = new_sums
                changed = True
                break
                
    return current, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_set = None
    best_size = float('inf')
    
    # 1. Initial run with ascending order
    order_asc = list(range(1, N + 1))
    s_asc, sums_asc = _construct_maximal(order_asc)
    # Verify maximality (should be true by construction)
    if _is_maximal(s_asc, sums_asc):
        s_asc_red, _ = _reduce_maximal(s_asc, sums_asc, deadline, rng)
        best_set = s_asc_red
        best_size = len(best_set)
    
    # 2. Randomized trials
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        s_rand, sums_rand = _construct_maximal(order)
        
        # Optional: Skip reduction if current size is already much larger than best?
        # No, reduction can be significant.
        
        s_red, _ = _reduce_maximal(s_rand, sums_rand, deadline, rng)
        
        if len(s_red) < best_size:
            best_size = len(s_red)
            best_set = s_red
            
    if best_set is None:
        # Fallback
        return _construct_maximal(list(range(1, N+1)))[0]
        
    return best_set
