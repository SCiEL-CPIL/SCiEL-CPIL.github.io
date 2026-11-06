"""
Program for large reciprocal-subsum-free sets using integer-scaled backtracking.
This version refactors the subset sum check and search heuristics for performance.
"""
import time
from math import gcd
from functools import reduce
import random

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return abs(a * b) // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _check_subsum(target, coeffs):
    """
    Check if target can be formed by a sum of a non-empty subset of coeffs.
    coeffs are positive integers.
    Uses Meet-in-the-Middle.
    Returns True if a valid non-empty subset sum equals target.
    """
    if not coeffs or target <= 0:
        return False
    
    # Filter coeffs to those <= target
    filtered = [c for c in coeffs if c <= target]
    if not filtered:
        return False
        
    if sum(filtered) < target:
        return False
        
    n = len(filtered)
    if n == 0:
        return False
        
    half = n // 2
    left = filtered[:half]
    right = filtered[half:]
    
    # Generate subset sums for left
    left_sums = {0}
    for val in left:
        current = list(left_sums)
        for s in current:
            new_s = s + val
            if new_s <= target:
                left_sums.add(new_s)
    
    # Generate subset sums for right
    right_sums = {0}
    for val in right:
        current = list(right_sums)
        for s in current:
            new_s = s + val
            if new_s <= target:
                right_sums.add(new_s)
    
    # Check for match
    # We need a non-empty subset. 
    # Since target > 0, any sum equal to target must come from at least one element.
    # So we just need to check if target is in the sum set.
    # The sum set includes 0 (empty sum). If target > 0, target != 0.
    # So if target is in left_sums or right_sums or any combination, it's valid.
    
    # Iterate over smaller set for efficiency
    if len(left_sums) < len(right_sums):
        for ls in left_sums:
            if target - ls in right_sums:
                return True
    else:
        for rs in right_sums:
            if target - rs in left_sums:
                return True
                
    return False

def _can_add(k, selected_list):
    """
    Check if k can be added to selected_list without violating constraints.
    """
    if not selected_list:
        return True
        
    L = _lcm_list(selected_list + [k])
    
    # Check 1: Is 1/k a subset sum of {1/y for y in selected_list}?
    # Equivalent to: Is L/k a subset sum of {L/y for y in selected_list}?
    target_k = L // k
    coeffs_k = [L // y for y in selected_list]
    if _check_subsum(target_k, coeffs_k):
        return False
        
    # Check 2: For each x in selected_list, is (1/x - 1/k) a subset sum of {1/y for y in selected_list, y != x}?
    # This checks if 1/x can be represented as 1/k + sum(1/y).
    # Equivalent to: Is L * (1/x - 1/k) a subset sum of {L/y for y in selected_list, y != x}?
    # L * (1/x - 1/k) = L * (k - x) / (x * k)
    
    for x in selected_list:
        if k <= x:
            # If k <= x, then 1/x <= 1/k, so 1/x - 1/k <= 0.
            # Since all coeffs are positive, the sum cannot be <= 0 unless it's 0 (empty set).
            # But the target is non-zero if k != x. If k == x, we are adding a duplicate, which is not allowed in a set context usually, 
            # but here k is not in selected_list. So k != x.
            # Thus target is negative or zero. Since target > 0 required for subsum check, this is impossible.
            continue
            
        val = L * (k - x) // (x * k)
        coeffs_x = [L // y for y in selected_list if y != x]
        
        if _check_subsum(val, coeffs_x):
            return False
            
    return True

def _validate_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    """
    S_list = list(S)
    for x in S_list:
        others = [y for y in S_list if y != x]
        if not others:
            continue
        L = _lcm_list([x] + others)
        target = L // x
        coeffs = [L // y for y in others]
        if _check_subsum(target, coeffs):
            return False
    return True

def _try_additions(S):
    """
    Try to add any element from 1..24 not in S.
    Returns (new_set, improved_flag).
    """
    S = set(S)
    for c in range(1, 25):
        if c not in S:
            if _can_add(c, list(S)):
                S.add(c)
                return S, True
    return S, False

def _try_swaps(S):
    """
    Try 1-2 swaps (remove 1, add 2) to increase cardinality.
    """
    S = set(S)
    candidates = [c for c in range(1, 25) if c not in S]
    if not candidates:
        return S, False
        
    for x in list(S):
        S_temp = S - {x}
        added = 0
        new_S = set(S_temp)
        for c in candidates:
            if _can_add(c, list(new_S)):
                new_S.add(c)
                added += 1
                if added > 1:
                    # We gained size
                    return new_S, True
        # If added == 1, size is same. We don't accept size-preserving swaps in this simple version
        # to avoid infinite loops or degrading quality, unless we add a "quality" metric.
        
    return S, False

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Multiple random restarts with greedy construction, followed by local search.
    """
    best_set = set()
    best_size = 0
    
    # Use a fixed seed for reproducibility
    random.seed(1234) # Different seed from previous version
    
    # Phase 1: Greedy construction with multiple orderings
    orderings = []
    orderings.append(list(range(24, 0, -1))) # Descending
    orderings.append(list(range(1, 25)))     # Ascending
    
    # Add random shuffles
    num_shuffles = 40
    for _ in range(num_shuffles):
        order = list(range(1, 25))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if _time_remaining() < 5.0:
            break
            
        S_temp = []
        for c in order:
            if _can_add(c, S_temp):
                S_temp.append(c)
                
        if len(S_temp) > best_size:
            best_set = set(S_temp)
            best_size = len(S_temp)
            
    # Phase 2: Local search for additions
    improved = True
    while improved and _time_remaining() > 5.0:
        best_set, improved = _try_additions(best_set)
        best_size = len(best_set)
        
    # Phase 3: 1-2 Swaps
    improved = True
    while improved and _time_remaining() > 2.0:
        best_set, improved = _try_swaps(best_set)
        best_size = len(best_set)
        
    return list(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_validate_set(result)}")
