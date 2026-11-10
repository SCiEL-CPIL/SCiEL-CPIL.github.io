"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Keep a safety margin before the 60s limit
    deadline = start_time + 45.0
    
    # We process elements in decreasing order: 24, 23, ..., 1
    elements = list(range(24, 0, -1))
    n = len(elements)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recips = {x: Fraction(1, x) for x in elements}
    
    best_set = []
    
    def branch_and_bound(idx, current_set, sums):
        nonlocal best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: If even adding all remaining elements doesn't beat best, stop
        if len(current_set) + (n - idx) <= len(best_set):
            return
            
        if idx == n:
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return
        
        x = elements[idx]
        rx = recips[x]
        
        # Option 1: Try to include x
        # Condition: 1/x must not be a subset sum of reciprocals of already selected (larger) elements
        if rx not in sums:
            # Calculate new set of subset sums
            # new_sums contains all old sums, plus all old sums + rx
            # Using a set comprehension
            new_sums = set(sums)
            # Optimization: update in place? No, sums is used in the other branch.
            # But we can create the new set efficiently.
            # Since we are in a recursion, creating a new set is fine.
            new_sums.update(s + rx for s in sums)
            
            branch_and_bound(idx + 1, current_set + [x], new_sums)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, current_set, sums)

    # Initial state: empty set, sums = {0}
    branch_and_bound(0, [], {Fraction(0)})
    
    # Return a non-empty subset. If best_set is empty (should not happen), return [24]
    if not best_set:
        return [24]
        
    # The problem asks for a subset of [1, 24]. Order doesn't matter for the set,
    # but returning sorted is nice.
    return sorted(best_set)
# EVOLVE-BLOCK-END
