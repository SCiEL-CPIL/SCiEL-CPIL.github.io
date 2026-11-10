"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    denominators = list(range(2, 49))  # 2 through 48
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    # Suffix sums for pruning
    suffix_sum = [Fraction() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + fractions[i]
    
    best_sum = Fraction()
    best_indices = None
    
    # Greedy initialization for a good starting point
    def greedy():
        nonlocal best_sum, best_indices
        selected, total = [], Fraction()
        for i in range(n):
            if total + fractions[i] < 1:
                selected.append(i)
                total += fractions[i]
        if total > best_sum:
            best_sum = total
            best_indices = selected[:]
    
    greedy()
    
    # Branch and bound DFS
    def dfs(idx, current_sum, selected):
        nonlocal best_sum, best_indices
        if time.time() > deadline:
            return
        
        if current_sum > best_sum and current_sum < 1:
            best_sum = current_sum
            best_indices = selected.copy()
        
        if idx == n:
            return
        
        # Prune: can't improve
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        # Try including fractions[idx]
        new_sum = current_sum + fractions[idx]
        if new_sum < 1:
            selected.append(idx)
            dfs(idx + 1, new_sum, selected)
            selected.pop()
        
        # Try excluding fractions[idx]
        if current_sum + suffix_sum[idx + 1] > best_sum:
            dfs(idx + 1, current_sum, selected)
    
    dfs(0, Fraction(), [])
    
    if best_indices is None:
        # Fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    return best_indices
# EVOLVE-BLOCK-END
