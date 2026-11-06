"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

# Precompute LCM of 1 to 22
_LCM = reduce(lcm, range(1, 23))
# Precompute integer values for each i in 1..22
# vals[i-1] corresponds to 1/i
_VALS = [_LCM // i for i in range(1, 23)]
_N = 22

def _is_dissociated_subset(indices):
    """
    Check if the subset of _VALS corresponding to indices is dissociated.
    This is a helper for potential verification, but the main search
    builds the sums incrementally.
    """
    if not indices:
        return True
    vals = [_VALS[i] for i in indices]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_subset = []
    best_cardinality = 0
    
    # Backtracking search
    # We iterate through candidates 0 to 21
    # state: current_index, current_sums (set), current_subset (list)
    
    def backtrack(idx, current_sums, current_subset):
        nonlocal best_subset, best_cardinality
        
        # Check deadline
        if time.time() > deadline:
            return
        
        # Update best if we have a larger set
        if len(current_subset) > best_cardinality:
            best_cardinality = len(current_subset)
            best_subset = list(current_subset)
        
        # Pruning: if even taking all remaining elements doesn't beat best, stop
        if len(current_subset) + (_N - idx) <= best_cardinality:
            return
            
        if idx == _N:
            return
            
        v = _VALS[idx]
        
        # Option 1: Include element at idx
        # Generate new sums
        new_sums = set()
        valid = True
        for s in current_sums:
            ns = s + v
            # Check if ns collides with existing sums or other new sums
            if ns in current_sums or ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
            
        if valid:
            next_sums = current_sums | new_sums
            current_subset.append(idx)
            backtrack(idx + 1, next_sums, current_subset)
            current_subset.pop()
            
        # Option 2: Exclude element at idx
        backtrack(idx + 1, current_sums, current_subset)
        
    # Start with empty set, sums={0}
    backtrack(0, {0}, [])
    
    # Map indices back to values 1..22
    # Index 0 corresponds to 1, index 1 to 2, etc.
    result = [i + 1 for i in best_subset]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
