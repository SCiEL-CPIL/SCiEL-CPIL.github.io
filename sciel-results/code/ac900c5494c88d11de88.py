"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin against 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums for pruning
    # suffix[i] = sum of recips[i:]
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Iterative DFS to avoid recursion depth issues and allow time checks
    # Stack elements: (index, current_sum, selected_indices_tuple)
    # We use a list for selected to avoid tuple copying overhead, 
    # but we need to track state carefully. 
    # Given n=47, recursion is fine, but iterative is safer for time checks.
    # However, standard recursion with sys.setrecursionlimit is often faster in Python 
    # if depth is small. Depth is 47, so recursion is safe.
    # Let's use recursion for cleaner code, as depth is only 47.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(idx, current_sum, selected):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: If even adding all remaining doesn't beat best, stop
        # Note: We require STRICTLY BELOW 1.
        # If current_sum + suffix[idx] <= best_sum, we cannot improve.
        # Also, if current_sum >= 1, we are invalid, but we only add if < 1.
        if current_sum + suffix[idx] <= best_sum:
            return
            
        # Base case: no more items
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(selected)
            return
        
        # Option 1: Exclude item idx
        dfs(idx + 1, current_sum, selected)
        
        # Option 2: Include item idx
        new_sum = current_sum + recips[idx]
        if new_sum < 1:  # Must be strictly below 1
            selected.append(idx)
            dfs(idx + 1, new_sum, selected)
            selected.pop()
            
    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
