"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _find_max_dissociated_subset():
    # Values are L/i
    values = [_L // i for i in range(1, 23)]
    
    # Sort in descending order
    values_desc = sorted(values, reverse=True)
    
    n = 22
    
    # We use a global variable to store the best subset found
    # Or just return the first one found for a given K
    # Since we iterate K from 22 down, the first found is the max.
    
    def search_with_path(idx, count, diff_set, target_k, path):
        if count == target_k:
            return True
        
        # Pruning: not enough elements left to reach target_k
        if n - idx + count < target_k:
            return False
        
        v = values_desc[idx]
        
        # Try to include values[idx]
        # Collision if v is in diff_set
        if v not in diff_set:
            # Calculate new diff_set
            # Optimization: create a new set
            new_diff_set = set(diff_set)
            for d in diff_set:
                new_diff_set.add(d + v)
                new_diff_set.add(d - v)
            
            # Map value back to original number
            # v = L / a => a = L / v
            a = _L // v
            path.append(a)
            
            if search_with_path(idx + 1, count + 1, new_diff_set, target_k, path):
                return True
            
            path.pop()
        
        # Try to skip values[idx]
        if search_with_path(idx + 1, count, diff_set, target_k, path):
            return True
        
        return False

    # Try to find a subset of size K, starting from 22 down to 1
    for k in range(22, 0, -1):
        path = []
        if search_with_path(0, 0, {0}, k, path):
            return path
        
    # Fallback
    return [1]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # The search is fast, but we guard against unexpected slowness
    result = _find_max_dissociated_subset()
    
    return result

# EVOLVE-BLOCK-END
