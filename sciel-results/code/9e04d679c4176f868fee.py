import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_validity(sums_set):
    """
    Check if no two distinct sums in sums_set divide one another.
    sums_set is a set of integers.
    """
    if not sums_set:
        return True
    
    # Convert to list for sorting? Or just iterate.
    # For each x in sums, check if any multiple y = k*x is in sums.
    # We only need to check k >= 2.
    # To speed up, we can iterate over x and check multiples.
    # To avoid double checking, we can just check if ANY multiple exists.
    
    # Optimization: If we sort sums, we can break early?
    # But set membership is O(1).
    
    max_sum = max(sums_set)
    
    for x in sums_set:
        # Check multiples of x
        # If 2x > max_sum, no multiples.
        if x * 2 > max_sum:
            continue
        multiple = x * 2
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += x
    return True

def _get_subset_sums(elements):
    """
    Generate all non-empty subset sums.
    """
    n = len(elements)
    if n == 0:
        return set()
    
    sums = set()
    # Iterative addition
    # Start with {0}
    current_sums = {0}
    for e in elements:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + e)
        current_sums.update(new_sums)
    
    current_sums.discard(0)
    return current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_size = 0
    best_set = []
    
    # Try to find a large set using greedy/randomized approach
    # We will try to build a set of size K.
    # Let's try K from 10 up to 20.
    
    # Candidates: 1..256
    # To avoid small numbers dividing large ones, maybe start with larger numbers?
    # But small numbers are "dangerous" because they divide many things.
    # Large numbers are "safe" because they have few multiples.
    # So prefer larger numbers?
    
    # Let's try a greedy construction:
    # 1. Start with empty set.
    # 2. Candidate pool: 256 down to 1.
    # 3. Try to add candidates in some order.
    
    # Randomized restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        # Shuffle candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Maybe sort by some heuristic?
        # Let's just iterate through shuffled candidates.
        
        current_set = []
        current_sums = set()
        
        for c in candidates:
            if time.time() > deadline:
                break
                
            # Try adding c
            new_elements = current_set + [c]
            new_sums = _get_subset_sums(new_elements)
            
            # Check validity
            # We can check if new_sums is valid.
            # Note: if current_sums was valid, we only need to check:
            # 1. New sums (c + old_sums) against old_sums and themselves.
            # 2. But full check is safer and simple for small N.
            
            if _check_validity(new_sums):
                current_set = new_elements
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                
                # If we reached a good size, we can break early for this restart?
                # Or continue to see if we can get more.
                # Since we are adding one by one, we can continue.
        
        # End of restart

    return best_set

# EVOLVE-BLOCK-END
