"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Reciprocals as Fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Suffix sums: suffix[i] = sum of recips[i:]
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]
    
    # Greedy initial solution: add in order while sum < 1
    best_indices = []
    best_sum = Fraction(0)
    cur = Fraction(0)
    for i in range(n):
        if cur + recips[i] < 1:
            cur += recips[i]
            best_indices.append(i)
    best_sum = cur
    
    # Branch and bound
    # State: (position, current_sum, current_indices)
    # Try to include (maximize sum while < 1)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(pos, cur_sum, indices):
        nonlocal best_sum, best_indices
        
        # Check time
        if time.time() > DEADLINE:
            return
        
        # Pruning: even taking all remaining can't beat best
        if cur_sum + suffix[pos] <= best_sum:
            return
        
        if pos == n:
            return
        
        # Option 1: Include recips[pos] if it keeps us < 1
        new_sum = cur_sum + recips[pos]
        if new_sum < 1:
            indices.append(pos)
            branch_and_bound(pos + 1, new_sum, indices)
            indices.pop()
        
        # Option 2: Exclude recips[pos]
        # Pruning check for exclude branch
        if cur_sum + suffix[pos + 1] > best_sum:
            branch_and_bound(pos + 1, cur_sum, indices)
    
    branch_and_bound(0, Fraction(0), [])
    
    return best_indices

# EVOLVE-BLOCK-END
