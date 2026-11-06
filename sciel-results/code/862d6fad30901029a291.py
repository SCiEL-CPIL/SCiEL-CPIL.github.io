"""Program for large reciprocal-subsum-free sets using greedy construction with exact checks."""
import time
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_subset_sum(target, elements):
    """Check if target equals the sum of reciprocals of some non-empty subset of elements."""
    if not elements:
        return False
    # Use DFS with memoization on (index, remaining_target)
    # But Fraction as key is expensive. Instead, use a set of achievable sums.
    achievable = {Fraction(0)}
    for e in elements:
        r = Fraction(1, e)
        # Check if target - r is achievable (meaning we can form target by including e)
        if target - r in achievable:
            return True
        # Add new sums
        new_achievable = set()
        for s in achievable:
            new_s = s + r
            new_achievable.add(new_s)
        achievable |= new_achievable
        # Early termination: if target is achievable
        if target in achievable:
            return True
    return target in achievable


def _can_add(k, S):
    """Check if k can be added to S while maintaining reciprocal-subsum-freeness."""
    S_list = list(S)
    
    # Check 1: 1/k should not be a subset sum of reciprocals of S
    if _is_subset_sum(Fraction(1, k), S_list):
        return False
    
    # Check 2: For each x in S, 1/x should not be a subset sum of reciprocals of (S\{x}) ∪ {k}
    for i, x in enumerate(S_list):
        others = S_list[:i] + S_list[i+1:] + [k]
        if _is_subset_sum(Fraction(1, x), others):
            return False
    
    return True


def _is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    if not S:
        return True
    S_list = list(S)
    for i, x in enumerate(S_list):
        others = S_list[:i] + S_list[i+1:]
        if _is_subset_sum(Fraction(1, x), others):
            return False
    return True


def construct():
    """Construct a large reciprocal-subsum-free subset of [1,24]."""
    if _time_remaining() < 2.0:
        return list(range(13, 25))
    
    # Greedy descending construction
    best_set = set()
    for k in range(24, 0, -1):
        if _time_remaining() < 2.0:
            break
        if _can_add(k, best_set):
            best_set.add(k)
    
    # Try to add remaining elements in ascending order
    for k in range(1, 25):
        if _time_remaining() < 2.0:
            break
        if k not in best_set and _can_add(k, best_set):
            best_set.add(k)
    
    # Try random restarts with different orders
    import random
    random.seed(12345)
    for attempt in range(50):
        if _time_remaining() < 3.0:
            break
        trial_set = set()
        order = list(range(1, 25))
        random.shuffle(order)
        for k in order:
            if _can_add(k, trial_set):
                trial_set.add(k)
        if len(trial_set) > len(best_set):
            best_set = trial_set
    
    # Final verification
    if not _is_valid(best_set):
        return list(range(13, 25))
    
    return sorted(best_set)
