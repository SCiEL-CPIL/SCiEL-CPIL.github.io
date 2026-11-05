from fractions import Fraction
import time
import itertools
from math import gcd

_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_exists_int(target_int, elements_int):
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
        # Check if target can be formed by subset of reciprocals of others
        if _subset_sum_exists(target, others):
            return False
    return True

def _subset_sum_exists(target_frac, elements):
    """Check if target_frac can be formed by a non-empty subset of reciprocals of elements."""
    if not elements:
        return False
        
    x = target_frac.denominator
    l = x
    for e in elements:
        l = _lcm(l, e)
        
    target_int = l // x
    elements_int = [l // e for e in elements]
    
    return _subset_sum_exists_int(target_int, elements_int)

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

def _local_search(S):
    """Try local search: remove one, try to add others."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            for j in range(len(S)):
                if i == j:
                    continue
                # Remove S[i], try to add S[j] (j not in S)
                # Actually, let's try: remove S[i], then try to add any excluded element
                new_S = [x for k, x in enumerate(S) if k != i]
                # Try adding each excluded element
                for x in range(1, 25):
                    if x not in new_S:
                        candidate = new_S + [x]
                        if _is_valid_set(candidate):
                            if len(candidate) > len(S):
                                S = candidate
                                improved = True
                                break
                if improved:
                    break
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
    
    _BEST = best
    return list(_BEST)
