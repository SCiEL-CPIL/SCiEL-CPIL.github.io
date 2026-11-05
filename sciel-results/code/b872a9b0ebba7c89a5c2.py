from fractions import Fraction
import time
import itertools
from math import gcd
from functools import reduce

_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _subset_sum_exists_int_mitm(target_int, elements_int):
    """
    Meet-in-the-middle check for subset sum.
    Returns True if target_int can be formed by a non-empty subset of elements_int.
    """
    if not elements_int:
        return False
    
    n = len(elements_int)
    # Sort descending for better pruning in DFS fallback, but for MITM order doesn't matter much
    # However, we need to ensure we don't use the empty set.
    # If target_int == 0, it's always formable by empty set, but problem requires non-empty.
    # Since elements are positive, sum is positive. So target > 0 implies non-empty subset.
    if target_int <= 0:
        return False
        
    total = sum(elements_int)
    if total < target_int:
        return False

    # Split into two halves
    mid = n // 2
    left = elements_int[:mid]
    right = elements_int[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for i in range(1 << len(left)):
        s = 0
        for j in range(len(left)):
            if i & (1 << j):
                s += left[j]
        left_sums.add(s)
        
    # Generate all subset sums for right half
    right_sums = set()
    for i in range(1 << len(right)):
        s = 0
        for j in range(len(right)):
            if i & (1 << j):
                s += right[j]
        right_sums.add(s)
        
    # Check if there exists l in left_sums, r in right_sums such that l + r == target_int
    # And (l, r) != (0, 0)
    for l in left_sums:
        r = target_int - l
        if r in right_sums:
            if l != 0 or r != 0:
                return True
    return False

def _subset_sum_exists_int_dfs(target_int, elements_int):
    """
    Fallback DFS with pruning for small sets or when MITM is too slow/complex.
    """
    if not elements_int:
        return False
    elements_int = sorted(elements_int, reverse=True)
    total = sum(elements_int)
    if total < target_int:
        return False
    
    n = len(elements_int)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + elements_int[i]
        
    def dfs_opt(idx, acc):
        if acc == target_int:
            return True
        if idx == n:
            return False
        if acc > target_int:
            return False
        if acc + suffix_sums[idx] < target_int:
            return False
            
        if dfs_opt(idx + 1, acc + elements_int[idx]):
            return True
        if dfs_opt(idx + 1, acc):
            return True
        return False
        
    return dfs_opt(0, 0)

def _subset_sum_exists(target_frac, elements):
    """
    Check if target_frac can be formed by a non-empty subset of reciprocals of elements.
    """
    if not elements:
        return False
        
    x = target_frac.denominator
    l = x
    for e in elements:
        l = _lcm(l, e)
        
    target_int = l // x
    elements_int = [l // e for e in elements]
    
    n = len(elements_int)
    # Use MITM for larger sets, DFS for small ones
    # MITM is O(2^(n/2)), DFS is O(2^n) worst case but prunes well.
    # For n <= 12, DFS is often faster in Python due to overhead of set generation.
    # For n > 12, MITM is better.
    if n <= 12:
        return _subset_sum_exists_int_dfs(target_int, elements_int)
    else:
        return _subset_sum_exists_int_mitm(target_int, elements_int)

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 50.0
    _BEST = None

def _is_valid_set(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = sorted(S, reverse=True)
    for x in S_list:
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True

def _greedy_decreasing():
    selected = []
    for x in range(24, 0, -1):
        if _subset_sum_exists(Fraction(1, x), selected):
            continue
        else:
            selected.append(x)
    return selected

def _try_add_back(excluded, selected):
    """Try to add back excluded elements one by one."""
    selected = list(selected)
    for x in sorted(excluded):  # try smaller first (harder to satisfy)
        new_selected = selected + [x]
        if _is_valid_set(new_selected):
            selected = new_selected
    return selected

def _local_search_1_out_k_in(S):
    """
    Try to improve S by removing one element and adding multiple candidates.
    """
    S = list(S)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_set = set(S)
        B = [x for x in range(1, 25) if x not in S_set]
        
        # Try removing each y in S
        for y in S:
            if time.time() > _DEADLINE:
                return S
            S_temp = [x for x in S if x != y]
            S_temp_set = set(S_temp)
            
            candidates = []
            for x in B:
                # Check if x is valid with respect to S_temp
                if not _subset_sum_exists(Fraction(1, x), S_temp):
                    candidates.append(x)
                    
            if len(candidates) > 1:
                # We can add all candidates
                # Verify validity of the new set just in case (should be valid by proof)
                new_S = S_temp + candidates
                if _is_valid_set(new_S):
                    S = new_S
                    improved = True
                    break # Restart the while loop with the new S
                    
    return S

def _local_search_swap(S):
    """
    Try to improve S by swapping one element for another, or removing one and adding one.
    """
    S = list(S)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_set = set(S)
        B = [x for x in range(1, 25) if x not in S_set]
        
        for y in S:
            if time.time() > _DEADLINE:
                return S
            S_temp = [x for x in S if x != y]
            for x in B:
                if time.time() > _DEADLINE:
                    return S
                if x == y:
                    continue
                new_S = S_temp + [x]
                if _is_valid_set(new_S):
                    # If we can add x without removing y, we would have done it in add_back
                    # So this is effectively a swap or an improvement in structure
                    if len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                    # If same size, maybe we prefer smaller numbers? 
                    # Or just accept any valid change to escape local optima?
                    # Let's accept any valid change that allows further growth.
                    # Actually, if len is same, we might just be cycling.
                    # Let's only accept if it leads to a different set that might allow growth.
                    # For now, only accept if strictly larger.
            if improved:
                break
    return S

def construct():
    _init()
    
    # Start with decreasing greedy
    best = _greedy_decreasing()
    
    if not best:
        best = [24]
    
    # Try to add back excluded elements
    all_elements = set(range(1, 25))
    excluded = list(all_elements - set(best))
    best = _try_add_back(excluded, best)
    
    # Apply local search 1-out-k-in
    best = _local_search_1_out_k_in(best)
    
    # Apply local search swap
    best = _local_search_swap(best)
    
    _BEST = best
    return list(_BEST)
