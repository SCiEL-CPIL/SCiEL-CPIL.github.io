"""Optimized distinct subset sums with small largest element."""
# EVOLVE-BLOCK-START
import time
import random

def _all_subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if distinct."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _subset_sums_set(arr):
    """Return set of all subset sums, or None if not distinct."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                return None
            new_sums.add(ns)
        sums |= new_sums
    return sums

def _greedy_construct(n, deadline):
    """Greedy: build set one element at a time, picking smallest valid value."""
    subset_sums = {0}
    result = []
    for i in range(n):
        candidate = 1
        while True:
            valid = True
            new_sums = set()
            for s in subset_sums:
                ns = s + candidate
                if ns in subset_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                subset_sums |= new_sums
                result.append(candidate)
                break
            candidate += 1
            if time.time() > deadline:
                return result
    return result

def _coordinate_descent(arr, deadline):
    """Try to reduce each element to its minimum feasible value given the others."""
    arr = list(arr)
    n = len(arr)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old = arr[i]
            others = [arr[j] for j in range(n) if j != i]
            other_sums = _subset_sums_set(others)
            if other_sums is None:
                continue
            other_vals = set(others)
            best_v = None
            # Try from smallest
            for v in range(1, old):
                if v in other_vals:
                    continue
                shifted = set(s + v for s in other_sums)
                if not (shifted & other_sums):
                    best_v = v
                    break
            if best_v is not None and best_v < old:
                arr[i] = best_v
                improved = True
    return arr

def _try_kick(arr, deadline, kick_count=5):
    """Attempt to escape local minima by increasing an element and re-optimizing."""
    best_arr = arr
    best_max = max(arr)
    n = len(arr)
    
    for _ in range(kick_count):
        if time.time() > deadline:
            break
        # Pick a random element to increase
        idx = random.randint(0, n - 1)
        new_arr = arr[:]
        # Increase by 1, 2, or 3
        increment = random.randint(1, 3)
        new_arr[idx] += increment
        
        # Check if still valid (distinctness)
        if not _all_subset_sums_distinct(new_arr):
            continue
            
        # Run coordinate descent on the kicked version
        optimized = _coordinate_descent(new_arr, deadline)
        
        if max(optimized) < best_max:
            best_arr = optimized
            best_max = max(optimized)
            
    return best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    # Greedy construction
    best = _greedy_construct(12, deadline)
    if len(best) < 12:
        best = [1 << i for i in range(12)]

    # Coordinate descent to minimize each element
    best = _coordinate_descent(best, deadline)
    
    # Try to kick out of local minima
    if time.time() < deadline - 5:
        best = _try_kick(best, deadline)

    # Verify final result
    if not _all_subset_sums_distinct(best):
        best = [1 << i for i in range(12)]

    return best
# EVOLVE-BLOCK-END
