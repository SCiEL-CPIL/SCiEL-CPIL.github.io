"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    vals = [float(f) for f in fracs]
    
    # Suffix sums of float values for pruning
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals[i]
        
    # Initial solution using Greedy
    best_sum_frac = Fraction(0, 1)
    best_indices = []
    current_sum = Fraction(0, 1)
    for i in range(n):
        if current_sum + fracs[i] < 1:
            current_sum += fracs[i]
            best_indices.append(i)
    best_sum_frac = current_sum
            
    start_time = time.time()
    timeout = 55.0 # Leave 5s margin
    
    def dfs(i, current_sum_frac, current_indices):
        nonlocal best_sum_frac, best_indices, start_time, timeout
        
        # Time check
        if time.time() - start_time > timeout:
            return
            
        # Validity check (should be handled by caller, but safe)
        if current_sum_frac >= 1:
            return
            
        # Update best if better and non-empty
        if len(current_indices) > 0 and current_sum_frac > best_sum_frac:
            best_sum_frac = current_sum_frac
            best_indices = current_indices.copy()
            
        if i == n:
            return
            
        # Pruning using float approximation
        # If current sum + max possible remaining sum <= best sum (with small epsilon), prune
        # We use a small negative epsilon to be conservative (avoid pruning valid branches due to float error)
        if float(current_sum_frac) + suffix_sums[i] <= float(best_sum_frac) - 1e-12:
            return
            
        # Option 1: Include index i
        new_sum_frac = current_sum_frac + fracs[i]
        if new_sum_frac < 1:
            current_indices.append(i)
            dfs(i + 1, new_sum_frac, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude index i
        dfs(i + 1, current_sum_frac, current_indices)

    # Run DFS
    dfs(0, Fraction(0, 1), [])
    
    return best_indices
# EVOLVE-BLOCK-END
