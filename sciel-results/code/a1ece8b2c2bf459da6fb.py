import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _check_subset_sum(target_num, target_den, items):
    """
    Checks if target_num/target_den can be formed as a sum of a subset of items.
    items: list of (num, den) tuples.
    Returns True if possible.
    """
    n = len(items)
    if n == 0:
        return target_num == 0
    
    # Scale to integers
    all_dens = [target_den] + [d for _, d in items]
    lcm_val = all_dens[0]
    for d in all_dens[1:]:
        lcm_val = _lcm(lcm_val, d)
        
    target_scaled = target_num * (lcm_val // target_den)
    if target_scaled < 0:
        return False
    if target_scaled == 0:
        return True
        
    scaled_items = [num * (lcm_val // den) for num, den in items]
    
    # Meet in the middle
    mid = n // 2
    left_items = scaled_items[:mid]
    right_items = scaled_items[mid:]
    
    left_sums = {0}
    for item in left_items:
        new_sums = set()
        for s in left_sums:
            new_sums.add(s + item)
        left_sums.update(new_sums)
        
    if target_scaled in left_sums:
        return True
        
    right_sums = {0}
    for item in right_items:
        new_sums = set()
        for s in right_sums:
            new_sums.add(s + item)
        right_sums.update(new_sums)
        
    for l in left_sums:
        if (target_scaled - l) in right_sums:
            return True
            
    return False

def _is_valid_addition(current_set, candidate):
    """
    Checks if adding candidate to current_set maintains the subsum-free property.
    """
    # 1. Check if 1/candidate is a sum of reciprocals of current_set
    if current_set:
        items = [(1, x) for x in current_set]
        if _check_subset_sum(1, candidate, items):
            return False
            
    # 2. Check if for any x in current_set, 1/x is a sum of reciprocals of (current_set \ {x}) + {candidate}
    # This is equivalent to: 1/x - 1/candidate is a sum of reciprocals of (current_set \ {x})
    # Note: 1/x - 1/candidate = (candidate - x) / (x * candidate)
    for x in current_set:
        if x == candidate:
            continue # Should not happen if candidate not in set
        # If candidate < x, 1/x - 1/candidate is negative, cannot be sum of positive reciprocals
        if candidate < x:
            continue
            
        num = candidate - x
        den = x * candidate
        
        # Items are current_set excluding x
        others = [y for y in current_set if y != x]
        if not others:
            # If others is empty, sum is 0. Can num/den be 0? No, num>0.
            continue
            
        items = [(1, y) for y in others]
        if _check_subset_sum(num, den, items):
            return False
            
    return True

def _greedy_construct(order):
    """
    Greedily construct a set by adding elements in the given order if valid.
    """
    current_set = []
    for candidate in order:
        if _time_remaining() < 2.0:
            break
        if _is_valid_addition(current_set, candidate):
            current_set.append(candidate)
    return current_set

def _local_search_improve(initial_set):
    """
    Try to improve the set by swapping elements or adding new ones.
    """
    current_set = list(initial_set)
    improved = True
    while improved and _time_remaining() > 5.0:
        improved = False
        # Try adding any missing element
        for candidate in range(1, 25):
            if candidate in current_set:
                continue
            if _is_valid_addition(current_set, candidate):
                current_set.append(candidate)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping: remove one, add one
        for i in range(len(current_set)):
            if _time_remaining() < 2.0:
                break
            removed = current_set[i]
            remaining = [x for x in current_set if x != removed]
            
            for candidate in range(1, 25):
                if candidate in current_set:
                    continue
                if _is_valid_addition(remaining, candidate):
                    new_set = remaining + [candidate]
                    # We accept the swap if it allows adding more later or is just valid.
                    # Since we only care about cardinality, and size is same, 
                    # we might want to check if this swap enables further additions.
                    # For now, just take the first valid swap to change state.
                    current_set = new_set
                    improved = True
                    break
            if improved:
                break
                
    return current_set

def construct():
    best_set = []
    
    # 1. Greedy with different orders
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        list(range(13, 25)) + list(range(1, 13)), # Mixed
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(10):
        rand_order = list(range(1, 25))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if _time_remaining() < 10.0:
            break
        s = _greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            
    # 2. Local search improvement on the best greedy result
    if best_set:
        improved_set = _local_search_improve(best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    # 3. More extensive local search with multiple restarts if time permits
    # We can try to improve other greedy results too
    if _time_remaining() > 15.0:
        for order in orders[:5]: # Just the first few
            s = _greedy_construct(order)
            if len(s) < len(best_set):
                # Only improve if it has potential (size close to best)
                if len(s) >= len(best_set) - 2:
                    improved_s = _local_search_improve(s)
                    if len(improved_s) > len(best_set):
                        best_set = improved_s

    return sorted(best_set)
