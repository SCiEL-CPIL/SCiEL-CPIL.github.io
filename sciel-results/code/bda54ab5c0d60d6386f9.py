"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # We want to find the largest dissociated subset of [1, 22]
    # Use backtracking with pruning
    
    # Precompute integer values: L // a for each a in 1..22
    int_vals = {a: _L // a for a in range(1, 23)}
    
    # We'll try to build the set greedily with backtracking
    # Start with a good heuristic ordering: try smaller denominators first
    # but actually, for dissociated sets, we want to be careful
    
    # Let's do a systematic backtracking search
    # Order elements to try including them
    elements = list(range(1, 23))
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # seed fallback
    best_size = len(best_set)
    
    # Backtracking: we build the set incrementally
    # State: current set of elements, current set of subset sums (as integers)
    
    def backtrack(idx, current_elements, current_sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Update best if we found something larger
        if len(current_elements) > best_size:
            best_set = list(current_elements)
            best_size = len(current_elements)
        
        # Try to add remaining elements
        for i in range(idx, 22):
            if time.time() > deadline:
                return
            
            a = elements[i]
            v = int_vals[a]
            
            # Check if adding v preserves dissociation
            # New sums = current_sums union (current_sums + v)
            # Need: current_sums and (current_sums + v) to be disjoint
            # i.e., no s in current_sums such that s + v is also in current_sums
            # Equivalently: (current_sums + v) & current_sums == empty
            
            # Efficient check: for each s in current_sums, check if s+v in current_sums
            # But that's O(n). Better: check if (current_sums | (current_sums+v)) has size 2*len(current_sums)
            
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + v
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if not valid:
                continue
            
            # Add the element
            current_elements.append(a)
            current_sums = current_sums | new_sums
            
            backtrack(i + 1, current_elements, current_sums)
            
            # Remove
            current_elements.pop()
            current_sums = current_sums - new_sums
    
    # Start with empty set
    backtrack(0, [], {0})
    
    return best_set

# EVOLVE-BLOCK-END
