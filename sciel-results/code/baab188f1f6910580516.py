"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def _check_subset_sum(target, elements):
    """
    Check if target (integer) can be formed as a sum of a subset of elements (list of ints).
    Uses meet-in-the-middle.
    """
    n = len(elements)
    if n == 0:
        return target == 0
    
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]
    
    # Generate all subset sums for left
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    # Check if any sum in left_sums can be combined with a sum from right to make target
    # We need to check if (target - ls) is in right_sums for any ls in left_sums.
    # Generate right sums on the fly or store them.
    # Since n <= 24, 2^12 = 4096. Storing is fine.
    
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    # Check intersection
    for ls in left_sums:
        rem = target - ls
        if rem in right_sums:
            # We must ensure the subset is non-empty.
            # If target > 0, and all elements are positive, 
            # a sum of 0 only comes from the empty set.
            # If rem == 0 and ls == target, the subset from left is non-empty (since target > 0 and elements > 0).
            # If rem == target and ls == 0, the subset from right is non-empty.
            # If both 0, target=0, but target is sum of reciprocals of distinct ints >= 1, so target >= 1 (scaled).
            # So any match implies a non-empty subset.
            return True
            
    return False

def _is_safe_int(x, S):
    """
    Return True if 1/x is NOT equal to sum of reciprocals of any non-empty subset of S.
    S is a list/set of integers.
    """
    if not S:
        return True
        
    # Compute LCM of all elements in S and x
    # Actually, we just need LCM of S to scale 1/x?
    # Let L = LCM(S).
    # 1/x = L / (x * L / x)? No.
    # Let L = LCM([x] + list(S)).
    # Then 1/x = L/x.
    # 1/y = L/y for y in S.
    # We check if L/x is a subset sum of {L/y for y in S}.
    
    elems = list(S)
    if not elems:
        return True
        
    L = 1
    for y in elems:
        L = L * y // math.gcd(L, y)
    # Include x in LCM to ensure integer scaling? 
    # If we use L = LCM(S), then L/x might not be integer.
    # Let L = LCM([x] + list(S)).
    L = L * x // math.gcd(L, x)
    
    target = L // x
    sums = [L // y for y in elems]
    
    return not _check_subset_sum(target, sums)

def _is_valid_full(subset):
    """Check if the entire subset is reciprocal-subsum-free."""
    s = list(subset)
    for x in s:
        others = [y for y in s if y != x]
        if not _is_safe_int(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Strategy 1: Greedy construction from largest to smallest.
    selected = []
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe_int(i, selected):
            selected.append(i)
            
    best_set = set(selected)
    
    # Strategy 2: Local Search - Add excluded elements
    excluded = list(set(range(1, 25)) - best_set)
    
    # Sort excluded: larger numbers are safer (smaller reciprocals)
    excluded.sort(reverse=True)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded:
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            # Check if adding x keeps set valid
            new_set = best_set | {x}
            if _is_valid_full(new_set):
                best_set = new_set
                excluded.remove(x)
                improved = True
                # Restart loop to allow other additions that might become possible?
                # Or just continue.
                break # Break to re-evaluate excluded list state? 
                # Actually, just updating best_set and continuing is fine, 
                # but removing from excluded helps.
        
    # Strategy 3: Local Search - Swaps
    # Try to swap an element in best_set with one in excluded to potentially unlock additions.
    # This is less likely to increase size directly, but might.
    # Given time constraints, we might skip complex swaps if additions are fast.
    # But let's try a few.
    
    # Re-calculate excluded
    excluded = list(set(range(1, 25)) - best_set)
    excluded.sort(reverse=True)
    
    # Try to add again (in case previous break skipped some)
    for x in excluded:
        if time.time() > deadline:
            break
        if x not in best_set:
            new_set = best_set | {x}
            if _is_valid_full(new_set):
                best_set = new_set
    
    if not best_set:
        return [24]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
