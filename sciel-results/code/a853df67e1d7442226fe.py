import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums of reciprocals
    # suffix_sum[i] = sum(recips[i] + ... + recips[n-1])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]
    
    # Initialize best with a known good solution
    # Indices 0, 1, 5, 41 correspond to 2, 3, 7, 43
    # Sum = 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806
    init_indices = [0, 1, 5, 41]
    init_sum = sum(recips[i] for i in init_indices)
    if init_sum >= 1:
        # Fallback if logic changes, but 1805/1806 < 1
        init_sum = Fraction(0)
        init_indices = []
        
    best_sum = init_sum
    best_indices = init_indices
    
    # Branch and Bound
    # We use a stack to avoid recursion limits if necessary, but depth is only 47.
    # State: (index, current_sum, current_indices)
    # We iterate through indices 0 to n-1
    # At each index, we have two choices: include or exclude.
    
    # To improve efficiency, we can prune:
    # 1. If current_sum >= 1, prune.
    # 2. If current_sum + suffix_sum[index] <= best_sum, prune (can't improve).
    
    stack = [(0, Fraction(0), [])]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, curr_sum, curr_indices = stack.pop()
        
        if idx == n:
            if curr_sum < 1 and curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = curr_indices
            continue
        
        # Pruning 1: If adding the maximum possible remaining sum doesn't beat best, skip
        if curr_sum + suffix_sum[idx] <= best_sum:
            continue
            
        # Option 1: Exclude current denominator
        stack.append((idx + 1, curr_sum, curr_indices))
        
        # Option 2: Include current denominator
        new_sum = curr_sum + recips[idx]
        if new_sum < 1:
            stack.append((idx + 1, new_sum, curr_indices + [idx]))
        # If new_sum >= 1, we don't push, effectively pruning this branch
        
    return best_indices
