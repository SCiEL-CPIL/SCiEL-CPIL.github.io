import sys
import time
from math import gcd
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    # Values v_i = L // i
    # We want to select indices i such that {L//i for i in S} has distinct subset sums.
    # Let's work with the values directly.
    items = []
    for i in range(1, 23):
        items.append((L // i, i))
    
    # Sort items by value. 
    # Heuristic: Sorting by value might help with pruning if we track the sum range?
    # Actually, for distinct subset sums, the order of addition affects the intermediate set size.
    # Let's try sorting by value descending (largest first).
    items.sort(key=lambda x: x[0], reverse=True)
    
    max_time = 55.0 # Leave 5s buffer
    start_time = time.time()
    
    # Backtracking to find a dissociated set of size k
    def find_subset(k, items, idx=0, current_sums=None, current_indices=None):
        if current_sums is None:
            current_sums = {0}
        if current_indices is None:
            current_indices = []
            
        if len(current_indices) == k:
            return current_indices
        
        if idx >= len(items):
            return None
            
        if time.time() - start_time > max_time:
            return None
            
        # Pruning: if we don't have enough items left to reach size k
        if len(current_indices) + (len(items) - idx) < k:
            return None
            
        val, orig_idx = items[idx]
        
        # Option 1: Include items[idx]
        # Check if adding val creates a collision
        new_sums = set()
        collision = False
        for s in current_sums:
            ns = s + val
            if ns in current_sums or ns in new_sums:
                collision = True
                break
            new_sums.add(ns)
            
        if not collision:
            # Try including
            result = find_subset(k, items, idx + 1, current_sums | new_sums, current_indices + [orig_idx])
            if result is not None:
                return result
        
        # Option 2: Exclude items[idx]
        result = find_subset(k, items, idx + 1, current_sums, current_indices)
        if result is not None:
            return result
            
        return None

    # Try to find the largest k
    # We know from the seed that 9 is possible. Let's start high.
    # Max possible is 22, but likely much lower.
    # Let's try from 15 down to 1.
    for k in range(15, 0, -1):
        result = find_subset(k, items)
        if result is not None:
            return result
            
    # Fallback if backtracking fails or times out without finding a solution
    # This should rarely happen if the search space is well-pruned
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
