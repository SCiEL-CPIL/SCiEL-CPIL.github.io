import time
import random
import sys

sys.setrecursionlimit(10000)

# Deadline guard: 55s wall-clock (5s safety margin under 60s limit)
_DEADLINE = 55.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """
    Check that no two distinct values in sums divide one another.
    Optimized: Sort sums. For each sum s, check if any multiple 2s, 3s... is in sums.
    """
    if not sums:
        return True
    
    sum_set = set(sums)
    max_sum = max(sum_set)
    
    # Iterate over a sorted list to ensure consistent behavior, though set iteration works too
    # Sorting helps if we wanted early exit on large s (fewer multiples), but here we check all.
    # Actually, iterating over the set directly is fine.
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
    """
    chosen = []
    current_sums = set()
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        if x in chosen:
            continue
        
        # Fast pre-check: x itself is a sum.
        # 1. x must not divide any existing sum
        # 2. No existing sum must divide x
        compatible = True
        for s in current_sums:
            if x % s == 0 or s % x == 0:
                compatible = False
                break
        
        if not compatible:
            continue
            
        new_sums = _subset_sums_incremental(current_sums, x)
        
        # Full check on new sums
        if _check_no_divisibility(new_sums):
            chosen.append(x)
            current_sums = new_sums
            
    return chosen

def _local_search(initial_set, deadline_check=None):
    """
    Local search: try to add elements from a pool, or swap elements to improve size/validity.
    Since we want to maximize size, we primarily try to add.
    If we can't add, we might try swapping a small element for a larger one that allows additions?
    For now, simple addition-focused local search.
    """
    current_elements = list(initial_set)
    current_sums = set()
    
    # Recompute sums for current elements
    n = len(current_elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += current_elements[i]
        current_sums.add(s)
        
    if not _check_no_divisibility(current_sums):
        return current_elements # Invalid start, return as is
        
    pool = [x for x in range(1, 257) if x not in current_elements]
    
    improved = True
    while improved and deadline_check and deadline_check() > 1.0:
        improved = False
        # Try adding any element from pool
        # Shuffle pool to avoid bias in local search
        random.shuffle(pool)
        for x in pool:
            # Fast pre-check
            compatible = True
            for s in current_sums:
                if x % s == 0 or s % x == 0:
                    compatible = False
                    break
            if not compatible:
                continue
                
            new_sums = _subset_sums_incremental(current_sums, x)
            if _check_no_divisibility(new_sums):
                current_elements.append(x)
                current_sums = new_sums
                pool.remove(x)
                improved = True
                # Restart pool check since sums changed
                break
                
    return current_elements

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    best = []
    best_len = 0
    
    # Strategy 1: Deterministic Greedy (Descending)
    # Large numbers are less likely to divide smaller sums, and their multiples are out of range.
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
        
    # Strategy 2: Randomized Greedy Restarts
    # Use fixed seeds for reproducibility
    rng = random.Random(42)
    
    # Run until deadline
    while _remaining() > 2.0:
        # Shuffle pool
        random_pool = list(pool)
        rng.shuffle(random_pool)
        
        result = _greedy_construct(random_pool, deadline_check=_remaining)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
        # If we have a good set, try local search to expand it
        if len(result) > 0 and _remaining() > 3.0:
            # Local search on the best found so far (or this one if it's better)
            # To save time, only run local search if we've found something decent or if we have time
            if len(result) >= best_len:
                improved_result = _local_search(result, deadline_check=_remaining)
                if len(improved_result) > best_len:
                    best = improved_result
                    best_len = len(improved_result)
                    
    # Final validation
    if best and not _validate_full(best):
        # Fallback to a known valid small set
        best = [128, 192, 224]
        
    # Ensure Python ints
    return [int(x) for x in best]

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
