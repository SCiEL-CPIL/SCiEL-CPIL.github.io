"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _subset_sum_exists_mitm(values, target):
    """Check if any non-empty subset of values sums to target using meet-in-the-middle."""
    n = len(values)
    if n == 0:
        return target == 0
    
    # Quick checks
    total = sum(values)
    if total < target:
        return False
    if target == 0:
        return False  # We need non-empty subset, and all values are positive
    
    # If target equals any single element
    for v in values:
        if v == target:
            return True
    
    # Meet in the middle
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for i in range(1 << len(left)):
        s = 0
        for j in range(len(left)):
            if i & (1 << j):
                s += left[j]
        left_sums.add(s)
    
    # For each subset sum of right half, check if target - that sum is in left_sums
    for i in range(1 << len(right)):
        s_right = 0
        for j in range(len(right)):
            if i & (1 << j):
                s_right += right[j]
        
        # Check if we can form target - s_right from left half
        # But we need non-empty subset overall
        if s_right == target:
            # Right subset alone forms target (non-empty since i > 0)
            return True
        if s_right < target and (target - s_right) in left_sums:
            # Need to ensure the combined subset is non-empty
            # If s_right == 0, we need left part to be non-empty
            # If s_right > 0, subset is already non-empty
            if s_right > 0:
                return True
            else:
                # s_right == 0, need left part to be non-empty
                # Check if target can be formed by non-empty subset of left
                # We already know target != 0 (checked above)
                # And we check if target is in left_sums, but that includes empty subset (sum=0)
                # Since target > 0 and target != 0, if target in left_sums, it must be from non-empty subset
                if target in left_sums:
                    return True
    
    return False

def _is_safe(selected, x):
    """Check if adding x to selected is safe (no reciprocal subsum violation)."""
    if not selected:
        return True
    
    # Compute LCM of selected and x
    L = _lcm_list(selected)
    L2 = _lcm(L, x)
    
    # Target in integer terms: L2 // x
    target = L2 // x
    
    # Each selected element y contributes L2 // y
    # We need to check if any non-empty subset sums to target
    values = [L2 // y for y in selected]
    
    # Quick check: if sum of all < target, impossible
    if sum(values) < target:
        return True
    
    # Use meet-in-the-middle for subset sum check
    return not _subset_sum_exists_mitm(values, target)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Greedy: try elements from 24 down to 1
    # When adding element x, check if 1/x can be formed by subset of reciprocals of already-selected elements
    # Already-selected elements are all > x, so their reciprocals are all < 1/x
    
    selected = []  # list of selected elements (will be in descending order)
    rejected = []  # elements rejected during greedy
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(x)
            continue
        
        # Check: can 1/x be formed as sum of reciprocals of a non-empty subset of 'selected'?
        if _is_safe(selected, x):
            selected.append(x)
        else:
            rejected.append(x)
    
    # Local search: try to add back rejected elements by removing conflicting ones
    # For each rejected element, try adding it and see if we can remove one conflicting element
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in rejected[:]:
                if x in selected:
                    continue
                # Try adding x: check which elements in selected conflict with x
                # An element y conflicts with x if 1/y = sum of reciprocals of subset of (selected - {y}) ∪ {x}
                # Or 1/x = sum of reciprocals of subset of selected
                # We already know 1/x can be formed from selected, so x conflicts
                # But maybe we can remove one element y from selected such that:
                # 1. x is safe with respect to (selected - {y})
                # 2. y is safe with respect to (selected - {y}) ∪ {x}
                # 3. No other element z in (selected - {y}) has 1/z formed by subset of (selected - {y}) ∪ {x}
                
                # Simpler approach: try removing each y from selected, then check if x can be added
                for y in selected[:]:
                    if time.time() > deadline - 5:
                        break
                    
                    new_selected = [z for z in selected if z != y]
                    
                    # Check if x is safe with new_selected
                    if not _is_safe(new_selected, x):
                        continue
                    
                    # Check if y is safe with new_selected ∪ {x}
                    test_set = new_selected + [x]
                    if not _is_safe(test_set, y):
                        continue
                    
                    # Check all other elements in new_selected are still safe
                    valid = True
                    for z in new_selected:
                        if not _is_safe([w for w in test_set if w != z], z):
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        selected = new_selected + [x]
                        rejected.remove(x)
                        rejected.append(y)
                        improved = True
                        break
                if time.time() > deadline - 5:
                    break
    
    # Return as a list (order doesn't matter for the contract)
    if not selected:
        return [24]
    return list(selected)

# EVOLVE-BLOCK-END
