"""Program for large reciprocal-subsum-free sets using optimized greedy + local search."""
import time
import math
from itertools import combinations
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

# Precompute LCM of 1..24
_LCM_24 = 1
for i in range(1, 25):
    _LCM_24 = _LCM_24 * i // math.gcd(_LCM_24, i)

# Precompute scaled reciprocals
_V = [_LCM_24 // i for i in range(1, 25)] # V[i-1] = LCM/i


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    if sum(items) < target:
        return False
    
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    Uses precomputed _V.
    """
    if len(S) < 2:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    # Map element to its scaled reciprocal
    # S contains 1..24. _V is 0-indexed for 1..24.
    # So for x in S, value is _V[x-1]
    
    vals = [_V[x-1] for x in S_list]
    
    for i in range(n):
        target = vals[i]
        others = [v for j, v in enumerate(vals) if j != i]
        
        if _can_form_sum(target, others):
            return False
            
    return True


def _try_add_elements(current, candidates, max_add=3):
    """
    Try to add up to max_add elements from candidates to current while maintaining subsum-freeness.
    Returns the best extended set found.
    """
    if not candidates or not current:
        return current
    
    best = set(current)
    
    # Try adding 1 element
    for c in candidates:
        if _time_remaining() < 3.0:
            break
        new_set = best | {c}
        if _is_subsum_free(new_set):
            if len(new_set) > len(best):
                best = new_set
    
    # Try adding 2 elements
    if len(candidates) >= 2 and _time_remaining() > 5.0:
        for c1, c2 in combinations(candidates, 2):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    # Try adding 3 elements
    if len(candidates) >= 3 and _time_remaining() > 8.0:
        for c1, c2, c3 in combinations(candidates, 3):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2, c3}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    return best


def _swap_improve(current):
    """
    Try to improve current set by swapping out one element for one or more others.
    Returns the best set found.
    """
    if not current or _time_remaining() < 5.0:
        return current
    
    best = set(current)
    all_elements = set(range(1, 25))
    missing = all_elements - best
    
    if not missing:
        return best
    
    # Try swapping out one element for one or more missing elements
    for elem_to_remove in list(best):
        if _time_remaining() < 3.0:
            break
        
        base_without = best - {elem_to_remove}
        
        # Try adding one missing element
        for m in missing:
            if _time_remaining() < 2.0:
                break
            new_set = base_without | {m}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
        
        # Try adding two missing elements
        if len(missing) >= 2 and _time_remaining() > 5.0:
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 2.0:
                    break
                new_set = base_without | {m1, m2}
                if _is_subsum_free(new_set):
                    if len(new_set) > len(best):
                        best = new_set
    
    return best


def _greedy_descending():
    """Greedy construction from 24 down to 1."""
    current = set()
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_ascending():
    """Greedy construction from 1 up to 24."""
    current = set()
    for k in range(1, 25):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_random(seed_val):
    """Greedy construction with random ordering."""
    import random
    rng = random.Random(seed_val)
    elems = list(range(1, 25))
    rng.shuffle(elems)
    current = set()
    for k in elems:
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _local_search(current):
    """Apply local search improvements."""
    if not current:
        return current
    
    all_elements = set(range(1, 25))
    missing = all_elements - current
    
    if missing and _time_remaining() > 10.0:
        current = _try_add_elements(current, list(missing), max_add=3)
    
    if _time_remaining() > 15.0:
        current = _swap_improve(current)
    
    missing = all_elements - current
    if missing and _time_remaining() > 8.0:
        current = _try_add_elements(current, list(missing), max_add=2)
    
    return current


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    best = set()
    
    # Try multiple greedy strategies and keep the best
    strategies = [
        _greedy_descending,
        _greedy_ascending,
    ]
    
    for strategy in strategies:
        if _time_remaining() < 10.0:
            break
        candidate = strategy()
        if len(candidate) > len(best):
            best = set(candidate)
    
    # Try more random orderings
    for seed in range(10):
        if _time_remaining() < 10.0:
            break
        candidate = _greedy_random(seed)
        if len(candidate) > len(best):
            best = set(candidate)
    
    if not best:
        return [24]
    
    # Apply local search to the best candidate
    best = _local_search(best)
    
    # Try local search on other candidates if time permits
    all_candidates = [
        _greedy_descending(),
        _greedy_ascending(),
    ]
    for seed in range(5):
        all_candidates.append(_greedy_random(seed))
    
    for candidate in all_candidates:
        if _time_remaining() < 15.0:
            break
        if len(candidate) >= len(best):
            improved = _local_search(candidate)
            if len(improved) > len(best):
                best = improved
    
    return list(best)
