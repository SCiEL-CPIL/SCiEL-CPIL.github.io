import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _can_form_target(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals.
    vals is a list of integers.
    """
    if target == 0:
        return True
    if not vals:
        return False
    if target > sum(vals):
        return False
    
    # Meet in the middle
    n = len(vals)
    half = n // 2
    left = vals[:half]
    right = vals[half:]
    
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
        
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
        
    for s in left_sums:
        if target - s in right_sums:
            return True
    return False

def _is_valid_set(S):
    """
    Check if no element's reciprocal is the sum of reciprocals of a subset of the others.
    """
    S_list = list(S)
    n = len(S_list)
    if n <= 1:
        return True
        
    for i, x in enumerate(S_list):
        others = S_list[:i] + S_list[i+1:]
        if not others:
            continue
            
        # Calculate LCM of others
        try:
            L = reduce(_lcm, others, 1)
        except ZeroDivisionError:
            continue # Should not happen with 1..24
            
        # Check if L/x is integer
        if L % x != 0:
            continue
            
        target = L // x
        if target == 0:
            continue
            
        vals = [L // y for y in others]
        
        if _can_form_target(target, vals):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy construction starting from large numbers
    # This tends to produce valid sets because large denominators sum to small values slowly.
    # Try different starting points.
    
    # Initial guess: [13..24] is known to be valid (size 12)
    candidate = list(range(13, 25))
    if _is_valid_set(candidate):
        best = candidate[:]
        best_size = len(best)
        
    # Try to expand [13..24] by adding smaller numbers
    # We will try to add numbers 1..12 in various orders.
    
    # Order 1: Descending (12, 11, ..., 1)
    current = set(best)
    for x in range(12, 0, -1):
        if time.time() > deadline:
            break
        test_set = list(current | {x})
        if _is_valid_set(test_set):
            current = set(test_set)
            
    if len(current) > best_size:
        best = list(current)
        best_size = len(best)
        
    # Order 2: Ascending (1, 2, ..., 12) - less likely to work, but try
    current = set(best) # Reset to best found so far? Or start from seed?
    # Let's restart from the seed [13..24] for each order to be fair
    seed_set = set(range(13, 25))
    
    for x in range(1, 13):
        if time.time() > deadline:
            break
        test_set = list(seed_set | {x})
        if _is_valid_set(test_set):
            seed_set = set(test_set)
            
    if len(seed_set) > best_size:
        best = list(seed_set)
        best_size = len(best)

    # Strategy 2: Randomized Greedy
    # Start with a random subset of [1..24] and greedily add elements.
    # Or start empty and add elements in random order.
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        elements = list(range(1, 25))
        random.shuffle(elements)
        
        S = []
        for x in elements:
            test = S + [x]
            # Optimization: only check if adding x creates a violation involving x or others?
            # _is_valid_set checks all. It's O(N^2 * 2^(N/2)).
            # For N=20, this is slow.
            # We can optimize: when adding x, only check if x is subsumed by others OR if any other is subsumed by others+x.
            # But _is_valid_set is simpler. Let's rely on the deadline.
            if _is_valid_set(test):
                S = test
                
        if len(S) > best_size:
            best = S
            best_size = len(best)
            
    # Strategy 3: Local Search / Swapping
    # If best is size K, try to swap elements to allow adding more?
    # Actually, just trying to add elements to the best set found so far in random order.
    
    if best:
        current = set(best)
        missing = [x for x in range(1, 25) if x not in current]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            test_set = list(current | {x})
            if _is_valid_set(test_set):
                current = set(test_set)
                
        if len(current) > best_size:
            best = list(current)
            best_size = len(best)

    return best
