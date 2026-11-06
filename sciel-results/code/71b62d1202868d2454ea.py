"""Program for large reciprocal-subsum-free sets using integer-scaled backtracking."""
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

def _is_subset_sum_int(target, coeffs):
    """
    Check if target can be formed by a sum of a non-empty subset of coeffs.
    coeffs are positive integers.
    """
    if not coeffs:
        return False
    if target <= 0:
        return False
    
    # Quick check: if any single coeff equals target
    for c in coeffs:
        if c == target:
            return True
        # If c > target, it can't be part of a sum equal to target (since all positive)
        # We can filter out coeffs > target to reduce search space
    
    # Filter coeffs to those <= target
    filtered = [c for c in coeffs if c <= target]
    if not filtered:
        return False
        
    # If sum of all filtered < target, return False
    if sum(filtered) < target:
        return False
        
    # Sort descending to find large sums quickly? 
    # Actually, for DFS, sorting ascending allows better pruning if we track remaining sum?
    # Let's sort ascending.
    filtered.sort()
    n = len(filtered)
    
    # Precompute suffix sums
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + filtered[i]
        
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        
        # Pruning: If current_sum + suffix_sums[idx] < target, return False
        if current_sum + suffix_sums[idx] < target:
            return False
            
        # Option 1: Include filtered[idx]
        new_sum = current_sum + filtered[idx]
        if new_sum <= target:
            if dfs(idx + 1, new_sum):
                return True
                
        # Option 2: Exclude filtered[idx]
        if dfs(idx + 1, current_sum):
            return True
            
        return False
        
    return dfs(0, 0)

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    selected is a list of integers.
    """
    if not selected:
        return True
        
    # Compute LCM of selected and k
    L = _lcm_list(selected + [k])
    
    # Check 1: Is 1/k a subset sum of reciprocals of selected?
    # Equivalent to: Is L/k a subset sum of {L/y for y in selected}?
    target_k = L // k
    coeffs_k = [L // y for y in selected]
    if _is_subset_sum_int(target_k, coeffs_k):
        return False
        
    # Check 2: For each x in selected, is (1/x - 1/k) a subset sum of reciprocals of selected \ {x}?
    # Equivalent to: Is L(1/x - 1/k) a subset sum of {L/y for y in selected \ {x}}?
    # L(1/x - 1/k) = L * (k - x) / (x * k)
    # Let val = L * (k - x) // (x * k)
    # Note: If k < x, 1/x - 1/k < 0, so target is negative. Sum of positive coeffs cannot be negative.
    # So we only need to check if k > x (so 1/x > 1/k).
    
    for x in selected:
        if k <= x:
            continue # 1/x <= 1/k, so 1/x - 1/k <= 0, impossible for positive sum
            
        # Calculate target value
        # val = L * (k - x) / (x * k)
        # Since L is multiple of x and k, this is integer.
        val = L * (k - x) // (x * k)
        
        # Coeffs are {L/y for y in selected if y != x}
        coeffs_x = [L // y for y in selected if y != x]
        
        if _is_subset_sum_int(val, coeffs_x):
            return False
            
    return True

def _validate_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    """
    S = set(S)
    for x in S:
        others = list(S - {x})
        if not others:
            continue
        L = _lcm_list([x] + others)
        target = L // x
        coeffs = [L // y for y in others]
        if _is_subset_sum_int(target, coeffs):
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
    Try to swap an element in S with an element not in S to increase size or maintain size with better properties?
    Actually, swaps don't increase size. We want to increase size.
    Swaps are useful if we are stuck at a local maximum for additions, but removing an element
    might allow adding two new ones.
    This is a 1-2 exchange.
    """
    S = set(S)
    candidates = [c for c in range(1, 25) if c not in S]
    if not candidates:
        return S, False
        
    # For each x in S, try removing it and adding two candidates?
    # That's expensive.
    # Simpler: Try removing x, then try adding any candidate.
    # If we can add a candidate after removing x, we have same size.
    # If we can add TWO candidates after removing x, we have size+1.
    
    for x in list(S):
        S_temp = S - {x}
        added = 0
        new_S = set(S_temp)
        # Try to add as many candidates as possible
        for c in candidates:
            if _can_add(c, list(new_S)):
                new_S.add(c)
                added += 1
                if added > 1:
                    # We gained size
                    return new_S, True
        # If added == 1, size is same. We might want to keep the one that allows more future additions?
        # For now, we only accept if size increases.
        
    return S, False

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Multiple random restarts with greedy construction, followed by local search (additions and swaps).
    """
    best_set = set()
    best_size = 0
    
    # Use a fixed seed for reproducibility in evaluation, but vary order
    random.seed(42)
    
    # Phase 1: Try several different orderings (greedy heuristics)
    orderings = []
    orderings.append(list(range(24, 0, -1)))
    orderings.append(list(range(1, 25)))
    
    # Add a few random shuffles
    for _ in range(30):
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
            
    # Phase 2: Local search on the best set found
    # Try to add more elements if possible
    improved = True
    while improved and _time_remaining() > 5.0:
        best_set, improved = _try_additions(best_set)
        best_size = len(best_set)
        
    # Phase 3: Try 1-2 swaps (remove 1, add 2) to increase cardinality
    # This is a heuristic to escape local optima where no single addition is possible,
    # but removing one element allows adding two.
    improved = True
    while improved and _time_remaining() > 2.0:
        best_set, improved = _try_swaps(best_set)
        best_size = len(best_set)
        
    return list(best_set)

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_validate_set(result)}")
