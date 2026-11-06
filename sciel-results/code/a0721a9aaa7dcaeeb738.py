"""
Program for large reciprocal-subsum-free sets.
Uses integer arithmetic with LCM scaling for fast subset-sum checks.
Implements greedy construction with multiple random restarts.
"""
import time
import math
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    """Compute LCM of two integers."""
    return a * b // gcd(a, b)


def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result


def _can_form_sum_integer(target_int, items_int):
    """
    Check if target_int can be formed as a sum of a non-empty subset of items_int.
    All values are positive integers.
    Uses DFS with pruning based on suffix sums.
    """
    if not items_int:
        return target_int == 0
    
    total_sum = sum(items_int)
    if total_sum < target_int:
        return False
    if target_int <= 0:
        return False
        
    n = len(items_int)
    # Sort in descending order for better pruning
    items_sorted = sorted(items_int, reverse=True)
    
    # Precompute suffix sums
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]
        
    found = [False]
    
    def dfs(i, current_sum, is_non_empty):
        if found[0]:
            return
        if i == n:
            if current_sum == target_int and is_non_empty:
                found[0] = True
            return
        
        # Prune 1: Even if we take all remaining items, we can't reach target
        if current_sum + suffix_sums[i] < target_int:
            return
            
        # Prune 2: If current_sum already exceeds target (and items are positive)
        if current_sum > target_int:
            return

        # Option 1: Exclude items_sorted[i]
        dfs(i + 1, current_sum, is_non_empty)
        if found[0]:
            return
            
        # Option 2: Include items_sorted[i]
        new_sum = current_sum + items_sorted[i]
        if new_sum <= target_int:
            dfs(i + 1, new_sum, True)
            
    dfs(0, 0, False)
    return found[0]


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    selected: list of ints
    candidate: int
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any non-empty subset of selected
    if selected:
        all_items = selected + [candidate]
        lcm_val = _lcm_list(all_items)
        # 1/candidate = lcm_val / (lcm_val * candidate)
        target_int = lcm_val // candidate
        items_int = [lcm_val // x for x in selected]
        if _can_form_sum_integer(target_int, items_int):
            return False
    
    # Check for each element in selected: 1/x should not equal sum of reciprocals of any non-empty subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        all_items = others
        lcm_val = _lcm_list(all_items)
        target_int = lcm_val // x
        items_int = [lcm_val // y for y in others]
        if _can_form_sum_integer(target_int, items_int):
            return False
            
    return True


def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    trial_set = []
    for candidate in order:
        if _time_remaining() < 2.0:
            break
        if _is_subsum_free(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    """
    all_elements = list(range(1, 25))
    best_set = []
    
    # 1. Deterministic greedy: Large to Small
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements, reverse=True))
        if len(s) > len(best_set):
            best_set = s

    # 2. Deterministic greedy: Small to Large
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements))
        if len(s) > len(best_set):
            best_set = s
            
    # 3. Random restarts
    random.seed(42)
    
    for _ in range(200):
        if _time_remaining() < 10.0:
            break
            
        # Random order
        candidates = list(all_elements)
        random.shuffle(candidates)
        
        trial_set = _greedy_construct(candidates)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    # 4. Biased random restarts
    for _ in range(100):
        if _time_remaining() < 10.0:
            break
            
        candidates = list(all_elements)
        random.shuffle(candidates)
        trial_set = _greedy_construct(candidates)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
