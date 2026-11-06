"""Optimized distinct subset sums with small largest element."""
# EVOLVE-BLOCK-START
import time

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
            # Try to find the smallest value < old that keeps distinctness
            # We need: adding value v to the subset sums of others must not collide
            # Compute subset sums of others (must be distinct)
            other_sums = _subset_sums_set(others)
            if other_sums is None:
                continue
            # For each candidate v from 1 to old-1, check if {s+v for s in other_sums}
            # is disjoint from other_sums and internally consistent
            # Actually we need: the full set of sums = other_sums ∪ {s+v for s in other_sums}
            # must be all distinct, i.e., {s+v for s in other_sums} disjoint from other_sums
            # AND {s+v for s in other_sums} itself is always distinct (since s distinct and +v is bijection)
            # So just need {s+v for s in other_sums} ∩ other_sums = ∅
            # Also v must not equal any other element (distinctness of the integers)
            other_vals = set(others)
            best_v = None
            # Try from smallest
            for v in range(1, old):
                if v in other_vals:
                    continue
                # Check disjointness
                shifted = set(s + v for s in other_sums)
                if not (shifted & other_sums):
                    best_v = v
                    break
            if best_v is not None and best_v < old:
                arr[i] = best_v
                improved = True
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    # Greedy construction
    best = _greedy_construct(12, deadline)
    if len(best) < 12:
        best = [1 << i for i in range(12)]

    # Coordinate descent to minimize each element
    best = _coordinate_descent(best, deadline)

    # Verify final result
    if not _all_subset_sums_distinct(best):
        best = [1 << i for i in range(12)]

    return best
# EVOLVE-BLOCK-END
