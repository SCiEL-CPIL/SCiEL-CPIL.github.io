"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _is_valid(subset):
    """
    Check if the subset is reciprocal-subsum-free.
    For each element x in subset, 1/x != sum(1/y for y in subset \ {x})
    """
    vals = [Fraction(1, x) for x in subset]
    # Precompute sum of all reciprocals
    total_sum = sum(vals)
    
    for i, x in enumerate(subset):
        # Sum of reciprocals of OTHER elements
        sum_others = total_sum - vals[i]
        # Check if 1/x == sum_others
        if vals[i] == sum_others:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    # Candidate 1: Top 19 integers {6..24}
    base_set = list(range(6, 25))
    
    # Verify base set
    if not _is_valid(base_set):
        # Fallback to top 18 if top 19 is invalid (unlikely)
        base_set = list(range(7, 25))
        
    current_best = base_set[:]
    
    # Try to expand by adding remaining elements {1, 2, 3, 4, 5}
    # We want to maximize cardinality, so try to add as many as possible.
    # Order: Try adding smaller numbers first? Or larger?
    # Smaller numbers have larger reciprocals, so they are more likely to cause conflicts.
    # But we want to include them if possible.
    # Let's try adding 5, 4, 3, 2, 1 in that order.
    candidates_to_add = [5, 4, 3, 2, 1]
    
    current_set = current_best[:]
    
    for num in candidates_to_add:
        if time.time() > deadline:
            break
        test_set = current_set + [num]
        # Ensure uniqueness (should be fine as num not in current_set)
        if len(test_set) == len(set(test_set)):
            if _is_valid(test_set):
                current_set = test_set
                current_best = current_set[:]
    
    # If we have time, try a simple local search: swap elements
    # If current size is < 24, try swapping an element in current_set with one outside
    # to see if we can maintain size or increase it? 
    # Since we started with 19 and tried to add, if we are stuck at 19, 
    # maybe a different combination of 20 exists.
    
    # Let's try a randomized restart if we have time and haven't reached 24
    if len(current_best) < 24 and time.time() < deadline - 10:
        import random
        best_size = len(current_best)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Random subset of size 20
            # Try to construct a set of size 20
            # Pick 20 random numbers from 1..24
            trial_set = random.sample(range(1, 25), 20)
            if _is_valid(trial_set):
                if len(trial_set) > best_size:
                    current_best = trial_set
                    best_size = len(trial_set)
                    # Try to expand this new best
                    remaining = [x for x in range(1, 25) if x not in current_best]
                    for num in remaining:
                        if time.time() > deadline - 2:
                            break
                        test_set = current_best + [num]
                        if _is_valid(test_set):
                            current_best = test_set
                            best_size = len(current_best)
    
    return current_best
# EVOLVE-BLOCK-END
