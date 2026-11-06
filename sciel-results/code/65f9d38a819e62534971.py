"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit
    
    # Candidates: 2 to 256
    candidates = list(range(2, 257))
    N = len(candidates)
    
    # Sort by value ascending (2, 3, 4...) which means weight descending (1/2, 1/3...)
    # This ordering helps B&B prune effectively as small numbers have high weight and many conflicts.
    
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if lcm(candidates[i], candidates[j]) <= 256
    conflict_mask = [0] * N
    for i in range(N):
        ci = candidates[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if lcm(ci, cj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Weights (reciprocals)
    weights = [1.0 / c for c in candidates]
    
    # Suffix sums of weights for loose upper bound
    suffix_w = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_w[i] = suffix_w[i + 1] + weights[i]
        
    best_sum = 0.0
    best_set = []
    
    # Update best solution
    def update_best(current_sum, selected_indices):
        nonlocal best_sum, best_set
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = selected_indices[:]
            
    # 1. Initial Greedy Solution
    # Greedy: Pick the first available (highest weight, lowest value)
    forbidden = 0
    current_sum = 0.0
    selected = []
    for i in range(N):
        if (forbidden >> i) & 1:
            continue
        # Select i
        selected.append(i)
        current_sum += weights[i]
        forbidden |= conflict_mask[i]
    update_best(current_sum, selected)
    
    # 2. Branch and Bound
    def solve(i, current_sum, forbidden_mask, selected_list):
        nonlocal best_sum, best_set
        
        # Time guard
        if time.time() > deadline:
            return
            
        # Loose Upper Bound
        if current_sum + suffix_w[i] <= best_sum:
            return
            
        # Tight Upper Bound
        # Sum of weights of all j >= i that are not forbidden
        ub_sum = 0.0
        for j in range(i, N):
            if not (forbidden_mask >> j) & 1:
                ub_sum += weights[j]
                
        if current_sum + ub_sum <= best_sum:
            return
            
        if i == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected_list[:]
            return
        
        # Option 1: Select i (if not forbidden)
        # We try Select first because we sorted by weight descending, 
        # so earlier indices have higher weights.
        if not (forbidden_mask >> i) & 1:
            new_forbidden = forbidden_mask | conflict_mask[i]
            new_sum = current_sum + weights[i]
            selected_list.append(i)
            solve(i + 1, new_sum, new_forbidden, selected_list)
            selected_list.pop()
            
        # Option 2: Skip i
        solve(i + 1, current_sum, forbidden_mask, selected_list)
        
    solve(0, 0.0, 0, [])
    
    # Construct result
    result = [candidates[i] for i in best_set]
    
    # Ensure non-empty
    if not result:
        result = [256]
        
    return result
# EVOLVE-BLOCK-END
