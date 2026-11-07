import time
import random
import sys

sys.setrecursionlimit(10000)

# Deadline guard: 85s wall-clock (5s safety margin under 90s limit)
_DEADLINE = 85.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """
    Check that no two distinct values in sums divide one another.
    """
    if not sums:
        return True
    
    sum_set = set(sums)
    max_sum = max(sum_set)
    
    for s in sum_set:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
            
    return True

def _subset_sums_incremental(current_sums, new_element):
    """
    Given a set of existing non-empty subset sums, add a new element.
    Returns the new set of sums (existing + existing+new + {new}).
    """
    new_sums = set(current_sums)
    new_sums.add(new_element)
    for s in current_sums:
        new_sums.add(s + new_element)
    return new_sums

def _is_valid_addition(current_sums, x):
    """
    Efficiently check if adding x to a set with sums `current_sums`
    maintains the divisibility-free property.
    """
    if not current_sums:
        return True
        
    # 1. Check x vs current_sums
    for s in current_sums:
        if x % s == 0 or s % x == 0:
            return False
            
    # Pre-compute new_sums for efficient checking
    new_sums_list = [s + x for s in current_sums]
    new_sums_set = set(new_sums_list)
    max_new = max(new_sums_list) if new_sums_list else 0
    
    # 2. Check new_sums vs current_sums
    for ns in new_sums_list:
        for os in current_sums:
            if ns % os == 0 or os % ns == 0:
                return False
                
    # 3. Check new_sums vs new_sums
    for ns in new_sums_list:
        multiple = ns * 2
        while multiple <= max_new:
            if multiple in new_sums_set:
                return False
            multiple += ns
            
    return True

def _validate_full(elements):
    """
    Full validation: distinct, in range, all non-empty subset sums no divisibility.
    """
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    if not _check_no_divisibility(sums):
        return False
    return True

def _greedy_construct(candidates, deadline_check=None):
    """
    Greedy: try adding candidates in given order, keeping only valid additions.
    Uses _is_valid_addition for speed.
    """
    chosen = []
    current_sums = set()
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        if x in chosen:
            continue
        
        # Use fast check
        if _is_valid_addition(current_sums, x):
            chosen.append(x)
            current_sums = _subset_sums_incremental(current_sums, x)
            
    return chosen

def _local_search(initial_set, deadline_check=None):
    """
    Local search: try to add elements. If stuck, try swaps.
    """
    current_elements = list(initial_set)
    current_sums = set()
    
    # Recompute sums
    n = len(current_elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += current_elements[i]
        current_sums.add(s)
        
    if not _check_no_divisibility(current_sums):
        return current_elements
        
    pool = [x for x in range(1, 257) if x not in current_elements]
    
    improved = True
    while improved and deadline_check and deadline_check() > 1.0:
        improved = False
        
        # Phase 1: Try to add
        random.shuffle(pool)
        for x in pool:
            if _is_valid_addition(current_sums, x):
                current_elements.append(x)
                current_sums = _subset_sums_incremental(current_sums, x)
                pool.remove(x)
                improved = True
                break
                
        if improved:
            continue
            
        # Phase 2: Try to swap
        if len(current_elements) > 0:
            for _ in range(5): # Try 5 random swaps
                if deadline_check and deadline_check() < 1.0:
                    break
                idx = random.randint(0, len(current_elements) - 1)
                u = current_elements[idx]
                
                # Remove u
                temp_elements = [e for i, e in enumerate(current_elements) if i != idx]
                temp_sums = set()
                n_temp = len(temp_elements)
                for mask in range(1, 1 << n_temp):
                    s = 0
                    for i in range(n_temp):
                        if mask & (1 << i):
                            s += temp_elements[i]
                    temp_sums.add(s)
                    
                # Try to add any v from pool to temp_elements
                found_swap = False
                random.shuffle(pool)
                for v in pool:
                    if v == u: continue
                    if _is_valid_addition(temp_sums, v):
                        # Accept swap
                        current_elements = temp_elements + [v]
                        current_sums = _subset_sums_incremental(temp_sums, v)
                        pool.remove(v)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
                    
    return current_elements

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    best = []
    best_len = 0
    
    # Strategy 1: Deterministic Greedy (Descending)
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
        
    # Strategy 2: Randomized Greedy Restarts
    rng = random.Random(42)
    
    # Run until deadline
    while _remaining() > 2.0:
        random_pool = list(pool)
        rng.shuffle(random_pool)
        
        result = _greedy_construct(random_pool, deadline_check=_remaining)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
        # Local search on the best found so far (or this one if it's better)
        if len(result) > 0 and _remaining() > 3.0:
            if len(result) >= best_len:
                improved_result = _local_search(result, deadline_check=_remaining)
                if len(improved_result) > best_len:
                    best = improved_result
                    best_len = len(improved_result)
                    
    # Final validation
    if best and not _validate_full(best):
        best = [128, 192, 224]
        
    # Ensure Python ints
    return [int(x) for x in best]

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
