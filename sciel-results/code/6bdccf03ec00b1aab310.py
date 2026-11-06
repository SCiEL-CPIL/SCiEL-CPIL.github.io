"""Large reciprocal-subsum-free sets: fast greedy with integer subset-sum check."""
import time
import math

_START = time.time()
_DEADLINE = 85.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _lcm_all():
    L = 1
    for k in range(1, 25):
        L = L * k // math.gcd(L, k)
    return L


_L = _lcm_all()
_W = [_L // k for k in range(1, 25)]  # _W[i-1] = 1/i * L


def _subset_sum_possible(target, weights):
    """Check if target can be formed as a sum of a non-empty subset of weights.
    Uses meet-in-the-middle. weights is a list of positive integers."""
    n = len(weights)
    if n == 0:
        return False
    if target <= 0:
        return False
    if sum(weights) < target:
        return False
    
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    # Generate all subset sums of left
    left_sums = set()
    for mask in range(1 << len(left)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += left[b]
            mm ^= lb
        left_sums.add(s)
    
    # Check right subsets
    for mask in range(1 << len(right)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += right[b]
            mm ^= lb
        if s > target:
            continue
        if (target - s) in left_sums:
            # Ensure non-empty: if both masks are 0, sum is 0, but target > 0 so this won't happen
            return True
    return False


def _is_valid(selected):
    """Check if the set of selected elements (1-indexed) is reciprocal-subsum-free."""
    if len(selected) < 2:
        return True
    sel_indices = [x - 1 for x in selected]  # 0-based
    for i in sel_indices:
        target = _W[i]
        others = [_W[j] for j in sel_indices if j != i]
        if _subset_sum_possible(target, others):
            return False
    return True


def construct():
    # Greedy: try to add elements in an order that maximizes cardinality
    # Try multiple orderings and pick the best
    
    best_result = []
    
    # Ordering 1: ascending
    for order in [range(1, 25), range(24, 0, -1), list(range(1, 25))[::-1] + list(range(1, 25))]:
        if _time_left() < 5.0:
            break
        current = []
        for x in order:
            if x in current:
                continue
            candidate = current + [x]
            if _is_valid(candidate):
                current = candidate
        if len(current) > len(best_result):
            best_result = current
    
    # Local improvement: try to add any missing element
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for x in range(1, 25):
            if x in best_result:
                continue
            candidate = best_result + [x]
            if _is_valid(candidate):
                best_result = candidate
                improved = True
    
    if not best_result:
        best_result = [24]
    
    return best_result
