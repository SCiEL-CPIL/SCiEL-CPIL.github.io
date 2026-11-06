"""Large reciprocal-subsum-free sets: Multi-strategy greedy with local search."""
import time
import math
import random
import sys
from functools import lru_cache

_START = time.time()
_DEADLINE = 85.0

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

# Precompute weights: W[k] = LCM(1..24) // k
_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [0] * 25
for k in range(1, 25):
    _W[k] = _L // k

def _subset_sum_exists(target, weights_list):
    """Check if target can be formed by a subset sum of weights_list.
    Uses meet-in-the-middle with bitmask enumeration."""
    if target < 0:
        return False
    if target == 0:
        return True
    if not weights_list:
        return False
    
    n = len(weights_list)
    # Quick check: sum of all < target
    if sum(weights_list) < target:
        return False
    
    mid = n // 2
    left = weights_list[:mid]
    right = weights_list[mid:]
    
    # Generate all subset sums of left
    left_sums = set()
    for mask in range(1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        left_sums.add(s)
    
    # Check all subset sums of right
    for mask in range(1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if s > target:
            continue
        if (target - s) in left_sums:
            return True
    return False

def _is_valid_addition(k, selected_set):
    """Check if adding element k (1-based) to selected_set (set of 1-based) is valid.
    selected_set does NOT contain k."""
    if not selected_set:
        return True
    
    # Check 1: Is W[k] a subset sum of weights in selected_set?
    weights_S = [_W[x] for x in selected_set]
    if _subset_sum_exists(_W[k], weights_S):
        return False
    
    # Check 2: For each x in selected_set, is W[x] a subset sum of (selected_set - {x}) + {k}?
    for x in selected_set:
        # W[x] should not be subset sum of (S \ {x}) U {k}
        # This means: W[x] != subset sum of (S\{x}), and W[x] - W[k] != subset sum of (S\{x})
        # The first was already valid (by induction), so check the second:
        diff = _W[x] - _W[k]
        if diff >= 0:
            others = [y for y in selected_set if y != x]
            if others:
                weights_others = [_W[y] for y in others]
                if _subset_sum_exists(diff, weights_others):
                    return False
            else:
                if diff == 0:
                    return False
    
    return True

def _validate_full(S):
    """Full validation of set S (set of 1-based integers)."""
    for x in S:
        others = S - {x}
        if not others:
            continue
        weights_others = [_W[y] for y in others]
        if _subset_sum_exists(_W[x], weights_others):
            return False
    return True

def _greedy(order):
    """Greedy construction in given order. Returns set of 1-based ints."""
    selected = set()
    for k in order:
        if _time_left() < 2.0:
            break
        if _is_valid_addition(k, selected):
            selected.add(k)
    return selected

def _local_search(S):
    """Try to improve set S via additions and swaps. Returns improved set."""
    S = set(S)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        
        # Try to add any element not in S
        for c in range(1, 25):
            if c not in S:
                if _is_valid_addition(c, S):
                    S.add(c)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps: remove one from S, add one not in S
        for remove in list(S):
            if _time_left() < 5.0:
                return S
            S_temp = S - {remove}
            for add in range(1, 25):
                if add in S_temp or add == remove:
                    continue
                if _is_valid_addition(add, S_temp):
                    S_temp.add(add)
                    # Now try to add more
                    for extra in range(1, 25):
                        if extra not in S_temp:
                            if _is_valid_addition(extra, S_temp):
                                S_temp.add(extra)
                    if len(S_temp) > len(S):
                        S = S_temp
                        improved = True
                        break
            if improved:
                break
    
    return S

def construct():
    _init_weights()
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Descending greedy (24, 23, ..., 1)
    S1 = _greedy(range(24, 0, -1))
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    # Strategy 2: Ascending greedy (1, 2, ..., 24)
    if _time_left() > 10.0:
        S2 = _greedy(range(1, 25))
        if len(S2) > best_size:
            best_set = S2
            best_size = len(S2)
    
    # Strategy 3: Random restarts
    random.seed(42)
    for trial in range(20):
        if _time_left() < 5.0:
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S_rand = _greedy(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
    
    # Local search on best set
    if _time_left() > 5.0:
        best_set = _local_search(best_set)
        best_size = len(best_set)
    
    if not best_set:
        return [24]
    
    return sorted(best_set)

def _init_weights():
    pass  # Already initialized at module level

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result}")
    print(f"Valid: {_validate_full(set(result))}")
