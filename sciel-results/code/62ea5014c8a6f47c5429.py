"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from math import lcm
from functools import lru_cache
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    deadline = time.time() + 75  # Safety margin before 90s limit
    
    # Precompute LCM of 1..22 to convert reciprocals to integers
    # LCM(1..22) = 232792560
    D = 232792560
    
    # Values for each element 1..22
    # val[x] = D // x
    vals = {x: D // x for x in range(1, 23)}
    
    all_elements = list(range(1, 23))
    
    # We want to find the largest subset S such that all subset sums of {vals[x] for x in S} are distinct.
    
    def is_valid_addition(current_sums_set, x_val):
        """
        Check if adding an element with value x_val to the current set of sums
        creates any collision.
        """
        # Check if any (s + x_val) is in current_sums_set
        for s in current_sums_set:
            if s + x_val in current_sums_set:
                return False
        return True
    
    best_set = []
    best_size = 0
    
    # Try to find a set of size k for k from 22 down to 1
    # Using backtracking with multiple random orders
    
    import random
    
    for target_size in range(22, 0, -1):
        if time.time() > deadline:
            break
        if target_size <= best_size:
            break
        
        # Try multiple random orderings
        max_attempts = 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            if found:
                break
                
            # Create a shuffled order
            shuffled_elements = all_elements[:]
            if attempt == 0:
                # First attempt: deterministic order (largest values first for better pruning)
                shuffled_elements.sort(key=lambda x: -vals[x])
            else:
                random.shuffle(shuffled_elements)
            
            current = []
            current_sums = {0}
            
            def backtrack(start_idx, remaining):
                nonlocal found, current, current_sums, best_set, best_size
                if time.time() > deadline:
                    return True
                if found:
                    return True
                
                if remaining == 0:
                    best_set = list(current)
                    best_size = target_size
                    found = True
                    return True
                
                # Pruning: not enough elements left
                if len(shuffled_elements) - start_idx < remaining:
                    return False
                
                for i in range(start_idx, len(shuffled_elements) - remaining + 1):
                    if time.time() > deadline:
                        return True
                    
                    x = shuffled_elements[i]
                    x_val = vals[x]
                    
                    if is_valid_addition(current_sums, x_val):
                        # Add x
                        current.append(x)
                        # Update sums
                        new_sums = set(current_sums)
                        for s in current_sums:
                            new_sums.add(s + x_val)
                        
                        old_sums = current_sums
                        current_sums = new_sums
                        
                        if backtrack(i + 1, remaining - 1):
                            return True
                        
                        # Undo
                        current.pop()
                        current_sums = old_sums
                
                return False
            
            backtrack(0, target_size)
            if found:
                break
        
        if found:
            break
    
    if not best_set:
        # Fallback to a known valid set if search fails or times out
        # Size 9 is known to be valid based on hints
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
