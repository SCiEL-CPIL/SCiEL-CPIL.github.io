"""Program for large reciprocal-subsum-free sets using Branch-and-Bound search."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 88.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target_num, target_den, items):
    """
    Check if target_num/target_den can be formed as a sum of a subset of items.
    items is a list of (num, den) tuples representing fractions.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(items)
    if n == 0:
        return target_num == 0
    
    # Convert all items and target to a common denominator basis
    all_dens = [target_den] + [d for _, d in items]
    lcm_val = all_dens[0]
    for d in all_dens[1:]:
        lcm_val = _lcm(lcm_val, d)
    
    # Scale target
    target_scaled = target_num * (lcm_val // target_den)
    
    # Scale items
    scaled_items = []
    for num, den in items:
        scaled_items.append(num * (lcm_val // den))
    
    # Now check if target_scaled can be formed as a sum of a subset of scaled_items
    if target_scaled == 0:
        return True
    
    if target_scaled < 0:
        return False
    
    # Meet-in-the-middle
    mid = n // 2
    left_items = scaled_items[:mid]
    right_items = scaled_items[mid:]
    
    # Generate all subset sums for left half
    left_sums = set([0])
    for item in left_items:
        new_sums = set()
        for s in left_sums:
            new_sums.add(s + item)
        left_sums.update(new_sums)
    
    # Check if target is in left_sums
    if target_scaled in left_sums:
        return True
    
    # Generate all subset sums for right half and check
    right_sums = set([0])
    for item in right_items:
        new_sums = set()
        for s in right_sums:
            new_sums.add(s + item)
        right_sums.update(new_sums)
    
    # Check if there's a pair (l, r) such that l + r = target_scaled
    for l in left_sums:
        if (target_scaled - l) in right_sums:
            return True
    
    return False


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any subset of selected
    if selected:
        items = [(1, x) for x in selected]
        if _can_form_sum(1, candidate, items):
            return False
    
    # Check for each element in selected: 1/x should not equal sum of reciprocals of any subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        items = [(1, y) for y in others]
        if _can_form_sum(1, x, items):
            return False
    
    return True


def _branch_and_bound(elements, current_set, best_set, index):
    """
    Recursive branch-and-bound search.
    elements: list of candidate elements (sorted descending)
    current_set: list of elements currently in the set
    best_set: list of elements in the best set found so far (modified in place)
    index: current index in elements
    """
    # Pruning: if we can't beat the best size, return
    if len(current_set) + (len(elements) - index) <= len(best_set):
        return
    
    if _time_remaining() < 1.0:
        return

    if index == len(elements):
        if len(current_set) > len(best_set):
            best_set[:] = current_set[:]
        return

    candidate = elements[index]
    
    # Branch 1: Include candidate
    # Only try to include if it keeps the set subsum-free
    if _is_subsum_free(current_set, candidate):
        current_set.append(candidate)
        _branch_and_bound(elements, current_set, best_set, index + 1)
        current_set.pop()
        
    # Branch 2: Exclude candidate
    _branch_and_bound(elements, current_set, best_set, index + 1)


def _local_search_improve(current_set):
    """Try to improve the set via single swaps (remove one, add one)."""
    best = list(current_set)
    improved = True
    while improved and _time_remaining() > 8.0:
        improved = False
        for i in range(len(best)):
            if _time_remaining() < 4.0:
                break
            removed = best[i]
            remaining = [x for x in best if x != removed]
            
            # Try to add any element not in best
            for candidate in range(24, 0, -1):
                if candidate in best:
                    continue
                if _is_subsum_free(remaining, candidate):
                    new_set = remaining + [candidate]
                    if len(new_set) > len(best):
                        best = new_set
                        improved = True
                        break
            if improved:
                break
    return best


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses Branch-and-Bound with exact integer subsum checks.
    """
    all_elements = list(range(24, 0, -1))  # Descending order for better pruning
    best_set = []
    
    # Initial heuristic: Greedy to get a good lower bound for pruning
    greedy_set = []
    for candidate in all_elements:
        if _time_remaining() < 5.0:
            break
        if _is_subsum_free(greedy_set, candidate):
            greedy_set.append(candidate)
    
    if len(greedy_set) > len(best_set):
        best_set = greedy_set[:]

    # Run B&B
    try:
        _branch_and_bound(all_elements, [], best_set, 0)
    except RecursionError:
        pass

    # If we have time left, try a local search on the best set found
    if _time_remaining() > 5.0:
        improved = _local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    return sorted(best_set)
