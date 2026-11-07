"""Rewritten program: active search for large sets with divisibility-incomparable subset sums."""
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _all_subset_sums_ok(sums_set):
    """Check that no two distinct values in sums_set divide one another.
    sums_set is a set of positive integers (the nonempty subset sums).
    If any two distinct values a,b in sums_set satisfy a|b or b|a, return False.
    Also, duplicates are already removed by the set, but the task says equal sum values
    from different subsets make the set invalid. So we must check for duplicate sums too.
    """
    pass  # This function is superseded by _is_valid_set which handles both checks.

def _is_valid_set(elements):
    """Check if the given set of distinct integers in 1..256 satisfies:
    1. All elements are distinct ints in 1..256.
    2. No two distinct nonempty subset sums divide one another.
    3. No two different subsets have the same sum (this is implied by condition 2 
       as explained in the task, but let's verify: if two subsets have the same sum s,
       their union has sum 2s and s divides 2s, so condition 2 would be violated 
       if 2s is also a subset sum. But 2s is the sum of the union, which is a valid 
       nonempty subset sum. So condition 2 already implies no duplicate subset sums.)
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all nonempty subset sums
    subset_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        subset_sums.add(s)
    
    # Check divisibility: no two distinct values in subset_sums divide one another
    sorted_sums = sorted(subset_sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    
    return True

def _is_valid_set_fast(elements):
    """Faster validity check using incremental subset sum generation."""
    n = len(elements)
    if n == 0:
        return True
    
    # Use a set to track all subset sums
    sums = {0}
    for x in elements:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        sums.update(new_sums)
    
    # Remove 0 (empty subset)
    sums.discard(0)
    
    if not sums:
        return True
    
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    
    return True

def _greedy_construct(max_value=256, max_size=30, deadline=None):
    """Greedy construction: add elements one by one if they preserve validity."""
    current = []
    sums = {0}
    
    for candidate in range(1, max_value + 1):
        if deadline is not None and time.time() > deadline:
            break
        # Try adding candidate
        new_sums = set(sums)
        for s in list(sums):
            new_sums.add(s + candidate)
        new_sums.discard(0)
        
        # Check validity
        sorted_sums = sorted(new_sums)
        valid = True
        for i in range(len(sorted_sums)):
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % sorted_sums[i] == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current.append(candidate)
            sums = new_sums
    
    return current

def _backtrack_search(elements_so_far, candidates, max_value=256, max_size=30, deadline=None):
    """Backtracking search to find a larger valid set."""
    best = list(elements_so_far)
    
    if deadline is not None and time.time() > deadline:
        return best
    
    if not candidates:
        return best
    
    if len(elements_so_far) >= max_size:
        return best
    
    # Compute current subset sums
    sums = {0}
    for x in elements_so_far:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
    sums.discard(0)
    
    for i, candidate in enumerate(candidates):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try adding candidate
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + candidate)
        new_sums.discard(0)
        
        # Check validity
        sorted_sums = sorted(new_sums)
        valid = True
        for a in range(len(sorted_sums)):
            for b in range(a + 1, len(sorted_sums)):
                if sorted_sums[b] % sorted_sums[a] == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            new_elements = elements_so_far + [candidate]
            remaining = candidates[i + 1:]
            result = _backtrack_search(new_elements, remaining, max_value, max_size, deadline)
            if len(result) > len(best):
                best = result
    
    return best

def construct():
    """Construct a set of distinct integers in 1..256 with divisibility-incomparable subset sums."""
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Start with greedy
    best = _greedy_construct(256, 30, deadline)
    
    # Try to improve with backtracking from different starting points
    # Try with a subset of the greedy result
    if len(best) > 0 and time.time() < deadline - 5:
        # Try backtracking with the first few elements of the greedy result
        for start_size in range(1, min(len(best) + 1, 8)):
            if time.time() > deadline - 2:
                break
            initial = best[:start_size]
            remaining_candidates = [x for x in range(1, 257) if x not in initial]
            # Filter to only candidates greater than the max in initial to maintain order
            max_initial = max(initial) if initial else 0
            remaining_candidates = [x for x in remaining_candidates if x > max_initial]
            
            result = _backtrack_search(initial, remaining_candidates, 256, 30, deadline)
            if len(result) > len(best):
                best = result
    
    # Ensure we return a valid list of Python ints
    return list(best)

# EVOLVE-BLOCK-END
