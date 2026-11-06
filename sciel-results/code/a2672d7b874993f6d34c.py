import time
from fractions import Fraction

_START_TIME = None
_TIME_LIMIT = 85.0  # 90s limit, keep 5s buffer

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

# Precompute reciprocals for 1..24
_RECIPS = {i: Fraction(1, i) for i in range(1, 25)}

# Memoization cache for subset sum checks
# Key: (target, frozenset(items)) -> bool
_subset_sum_cache = {}

def _subset_sum_possible(target, items):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of 'items' (list of Fractions).
    items are all positive.
    Returns True if possible, False otherwise.
    """
    if target == 0:
        return True
    if not items:
        return False
    if target < 0:
        return False
    
    # Create a cache key using the sorted items and target
    items_sorted = tuple(sorted(items, key=lambda x: x.numerator * 1.0 / x.denominator, reverse=True))
    cache_key = (target, items_sorted)
    if cache_key in _subset_sum_cache:
        return _subset_sum_cache[cache_key]
    
    # Sort items descending to hit large sums quickly and prune
    items = list(items_sorted)
    
    n = len(items)
    # Precompute suffix sums for lower bound pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i]
        
    # If total sum < target, impossible
    if suffix_sums[0] < target:
        _subset_sum_cache[cache_key] = False
        return False
        
    # DFS with pruning
    def dfs(idx, current_sum):
        if _time_exceeded():
            return False
        if current_sum == target:
            return True
        if idx == n:
            return False
        if current_sum > target:
            return False
        # Prune if even taking all remaining items can't reach target
        if current_sum + suffix_sums[idx] < target:
            return False
            
        # Try taking items[idx]
        if dfs(idx + 1, current_sum + items[idx]):
            return True
        # Try skipping items[idx]
        if dfs(idx + 1, current_sum):
            return True
        return False
        
    result = dfs(0, Fraction(0))
    _subset_sum_cache[cache_key] = result
    return result

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    S is a list of selected integers.
    """
    if x in S:
        return False
        
    recip_x = _RECIPS[x]
    
    # Check 1: 1/x is not a subset sum of reciprocals of S
    if S:
        recips_S = [_RECIPS[y] for y in S]
        if _subset_sum_possible(recip_x, recips_S):
            return False
        
        # Check 2: For each y in S, 1/y is not a subset sum of S ∪ {x} \ {y}
        for y in S:
            if y < x:
                # 1/y > 1/x, so target is positive
                target = _RECIPS[y] - recip_x
                # Items are reciprocals of S excluding y and x
                items = [_RECIPS[z] for z in S if z != y and z != x]
                if _subset_sum_possible(target, items):
                    return False
            # If y > x, target is negative, safe.
            # If y == x, not possible.
            
    return True

def _greedy_initial():
    """
    Greedy strategy: Add elements from 24 down to 1.
    Larger elements are "safer" (smaller reciprocals, harder to sum to larger targets).
    """
    S = []
    for x in range(24, 0, -1):
        if _can_add(x, S):
            S.append(x)
    return S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Get a good initial solution
    best_set = _greedy_initial()
    
    # 2. Branch and Bound to find the true maximum
    # We search over subsets of {1, ..., 24}.
    
    def search(i, S):
        nonlocal best_set
        if _time_exceeded():
            return
            
        # Pruning: Even if we add all remaining elements, we can't beat best_set
        if len(S) + (24 - i) <= len(best_set):
            return
            
        if i == 24:
            if len(S) > len(best_set):
                best_set = list(S)
            return
            
        x = i + 1
        
        # Strategy: Try Include first if x is large (safer), Exclude first if x is small (more restrictive)
        # Large numbers have smaller reciprocals, so they're less likely to cause conflicts
        if x > 12:
            # Try Include first for large numbers
            if _can_add(x, S):
                S.append(x)
                search(i + 1, S)
                S.pop()
            # Then try Exclude
            search(i + 1, S)
        else:
            # Try Exclude first for small numbers
            search(i + 1, S)
            # Then try Include
            if _can_add(x, S):
                S.append(x)
                search(i + 1, S)
                S.pop()
            
    search(0, [])
    
    if not best_set:
        return [24]
        
    return best_set
