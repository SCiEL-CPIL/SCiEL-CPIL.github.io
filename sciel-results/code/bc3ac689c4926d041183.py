"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _is_valid_set(S):
    """
    Check if S (list of distinct ints) satisfies:
    1) All nonempty subset sums are distinct (no two different subsets have same sum).
    2) No two distinct nonempty subset sums divide one another.
    """
    if len(set(S)) != len(S):
        return False
    n = len(S)
    sums = set()
    # Generate all nonempty subset sums
    for r in range(1, n + 1):
        for comb in combinations(S, r):
            s = sum(comb)
            if s in sums:
                return False
            sums.add(s)
    # Check divisibility among distinct sums
    sums_list = sorted(sums)
    sum_set = set(sums_list)
    max_sum = sums_list[-1] if sums_list else 0
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _get_sums(S):
    """Compute all nonempty subset sums of S. Returns set or None if collision."""
    sums = set()
    n = len(S)
    for r in range(1, n + 1):
        for comb in combinations(S, r):
            s = sum(comb)
            if s in sums:
                return None
            sums.add(s)
    return sums

def _check_divisibility(sums_set):
    """Check if no two distinct values in sums_set divide one another."""
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1] if sums_list else 0
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Known valid set from seed
    best_set = [31, 32, 34]
    
    # Try to extend the known valid set greedily
    def extend_greedy(current_set, max_val=256):
        base_sums = _get_sums(current_set)
        if base_sums is None:
            return None
        
        for x in range(min(current_set), max_val + 1):
            if x in current_set:
                continue
            
            # Check if adding x creates duplicate sums
            new_sums = set(base_sums)
            valid_distinct = True
            for s in base_sums:
                ns = s + x
                if ns in new_sums:
                    valid_distinct = False
                    break
                new_sums.add(ns)
            
            if not valid_distinct:
                continue
            
            # Check divisibility
            if _check_divisibility(new_sums):
                return x
        
        return None

    # First, try to extend from the known set
    current = list(best_set)
    while time.time() < deadline - 10:
        x = extend_greedy(current)
        if x is None:
            break
        current.append(x)
        if len(current) >= 30:
            break
    
    if len(current) > len(best_set) and _is_valid_set(current):
        best_set = current

    # If we still have time, try random search for larger sets
    import random
    
    def random_search(k, max_val=256, iterations=1000):
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            S = sorted(random.sample(range(1, max_val + 1), k))
            if _is_valid_set(S):
                return S
        return None

    # Try to find sets larger than best_set
    for k in range(len(best_set) + 1, 15):
        if time.time() > deadline - 10:
            break
        S = random_search(k, iterations=200)
        if S and len(S) > len(best_set):
            best_set = S

    # Final validation
    if _is_valid_set(best_set):
        return list(best_set)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
