"""Program for large reciprocal-subsum-free sets using optimized greedy construction."""
import time
import math
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


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
    # We'll work with integers by scaling everything to the LCM of all denominators
    
    # First, get the LCM of all denominators in items and target_den
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
    # Quick check: if target_scaled is 0, return True (empty subset)
    if target_scaled == 0:
        return True
    
    # If target is negative, we can't form it with positive items
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
    For each element in selected, check if 1/element can be formed as sum of reciprocals of a subset of OTHER selected elements + candidate.
    Also check if 1/candidate can be formed as sum of reciprocals of a subset of selected.
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


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses greedy construction with meet-in-the-middle subset sum checks.
    """
    # Start with a greedy approach: try to add elements in a strategic order
    # Larger elements have smaller reciprocals, so they're less likely to form sums
    # Try adding elements from large to small first
    
    best_set = []
    
    # Try different starting strategies
    # Strategy 1: Add from large to small
    for candidate in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        if _is_subsum_free(best_set, candidate):
            best_set.append(candidate)
    
    # Now try to improve by removing some elements and adding others
    # Try all single removals and see if we can add a better element
    improved = True
    while improved and _time_remaining() > 10.0:
        improved = False
        current_set = list(best_set)
        
        for i in range(len(current_set)):
            if _time_remaining() < 5.0:
                break
            removed = current_set[i]
            remaining = [x for x in current_set if x != removed]
            
            # Try to add any element not in current_set
            for candidate in range(24, 0, -1):
                if candidate in current_set:
                    continue
                if _is_subsum_free(remaining, candidate):
                    new_set = remaining + [candidate]
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
    
    # Try random restarts with different orderings
    import random
    random.seed(42)
    
    for _ in range(50):
        if _time_remaining() < 10.0:
            break
        
        # Random order
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        
        trial_set = []
        for candidate in candidates:
            if _time_remaining() < 5.0:
                break
            if _is_subsum_free(trial_set, candidate):
                trial_set.append(candidate)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
    
    return sorted(best_set)
