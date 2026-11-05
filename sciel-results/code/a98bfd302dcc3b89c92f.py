import time
from math import lcm
from itertools import combinations

def _can_form_target_int(target_int, values_int, n_values):
    """
    Meet-in-the-middle: check if target_int can be formed as sum of a non-empty
    subset of values_int (all positive integers).
    """
    if n_values == 0:
        return False
    total = sum(values_int)
    if total < target_int:
        return False
    
    half = n_values // 2
    left = values_int[:half]
    right = values_int[half:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for mask in range(1, 1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        if s <= target_int:
            left_sums.add(s)
    
    # Generate all subset sums for right half and check
    for mask in range(1, 1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s > target_int:
            continue
        need = target_int - s
        if need == 0:
            return True
        if need in left_sums:
            return True
    
    # Also check if target can be formed using only left or only right
    if target_int in left_sums:
        return True
    
    # Check right-only subsets
    for mask in range(1, 1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s == target_int:
            return True
    
    return False

def _is_valid_set(S):
    S = sorted(S)
    n = len(S)
    # Compute LCM of all elements
    lcm_val = 1
    for x in S:
        lcm_val = lcm(lcm_val, x)
    
    for idx, x in enumerate(S):
        # Target: 1/x scaled by LCM = lcm_val // x
        target_int = lcm_val // x
        # Values: 1/y scaled by LCM = lcm_val // y for y != x
        values_int = [lcm_val // y for i, y in enumerate(S) if i != idx]
        if _can_form_target_int(target_int, values_int, n - 1):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85
    
    # Start with the known good set {13, 14, ..., 24} of size 12
    best = set(range(13, 25))
    
    # Try to add elements from 12 down to 1
    for c in range(12, 0, -1):
        if time.time() > deadline:
            break
        trial = best | {c}
        if _is_valid_set(trial):
            best = trial
    
    # Try to add elements in a different order if time permits
    if time.time() < deadline - 5:
        # Try adding elements not yet in best
        remaining = [c for c in range(1, 25) if c not in best]
        # Sort by some heuristic: smaller numbers first might be easier to add
        remaining.sort()
        for c in remaining:
            if time.time() > deadline:
                break
            trial = best | {c}
            if _is_valid_set(trial):
                best = trial
    
    return sorted(best)
