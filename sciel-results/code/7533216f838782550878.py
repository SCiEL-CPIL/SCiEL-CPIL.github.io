import time
import math
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a non-empty subset of items.
    Uses meet-in-the-middle.
    items are integers.
    """
    if target <= 0:
        return False # Reciprocals are positive, sum must be positive
    
    n = len(items)
    if n == 0:
        return False
    if sum(items) < target:
        return False
    
    # If target is 0, we'd need empty subset, but we exclude that.
    # Since items are positive, any non-empty subset has sum > 0.
    
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    # Generate subset sums for left
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    # Generate subset sums for right
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    # Check if any pair sums to target
    # We must ensure the subset is not empty.
    # If s_left == 0 and s_right == 0, sum is 0. Target > 0, so this won't match.
    # So any match implies non-empty subset.
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False

def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    """
    if len(S) < 2:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    # Calculate LCM of all elements in S
    L = 1
    for x in S_list:
        L = _lcm(L, x)
        
    # Convert reciprocals to integers: 1/x becomes L/x
    vals = [L // x for x in S_list]
    
    for i in range(n):
        target = vals[i]
        # Items available are all vals except vals[i]
        # We pass a list of others
        others = []
        for j in range(n):
            if j != i:
                others.append(vals[j])
                
        if _can_form_sum(target, others):
            return False
            
    return True

def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    current = set()
    
    # Greedy descending: try to add elements from 24 down to 1
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
            
    # Local Search / Refinement
    # Try to add any missing element
    # We do this in a loop because adding one might allow adding another?
    # No, adding an element only makes the set "tighter" (more constraints).
    # If we can't add k now, we can't add it later if we only add more elements.
    # So a single pass of "try to add any missing" is sufficient for additions.
    
    missing = [k for k in range(1, 25) if k not in current]
    for k in missing:
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
            # Note: we don't remove k from missing list explicitly, 
            # but since we only add, and we check validity, it's fine.
            # However, if we add k, the set changes, so previously "missing" 
            # elements are still missing.
            # But wait, if we add k, we should re-evaluate? No, adding k 
            # only adds constraints. It cannot make a previously invalid 
            # addition valid. So one pass is enough.
            
    # Optional: Try swaps to see if we can reach a different local maximum 
    # that allows more additions?
    # This is complex and might not yield gains quickly.
    # The greedy 24->1 is very strong for this specific problem structure.
    
    if not current:
        return [24]
        
    return list(current)
