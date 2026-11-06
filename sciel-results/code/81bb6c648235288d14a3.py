import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Precompute suffix sums of weights for the fractional knapsack bound
    # suffix_sum[i] = sum(weights[i:])
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    # Best solution tracking
    best_sum_weight = 0
    best_indices = []
    
    def update_best(sum_w, indices):
        nonlocal best_sum_weight, best_indices
        if sum_w > best_sum_weight:
            best_sum_weight = sum_w
            best_indices = list(indices)
            
    # DFS with Branch and Bound
    # We want to maximize sum(weights[i]) subject to sum(weights[i]) < L
    # This is equivalent to minimizing L - sum(weights[i])
    
    # Heuristic: "Take First"
    # At each step, try taking the current item first, then skipping it.
    # This helps find large sums early, raising the best_sum_weight threshold
    # and improving pruning.
    
    # Optimization: Use iterative DFS or careful recursion to avoid stack overflow
    # and overhead. Also, we can use a tighter bound if possible, but suffix sum
    # is already standard.
    
    # Key improvement from inspiration: 
    # The current program is already quite strong. 
    # Let's ensure we are not wasting time on invalid branches.
    # The condition `current_sum + weights[idx] < L` is strict.
    
    # Let's add a small optimization: 
    # If we are at the last item, just check directly.
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum_weight, best_indices
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning: If even taking all remaining items doesn't beat the best, prune
        # Upper bound = current_sum + suffix_sum[idx]
        # If current_sum + suffix_sum[idx] <= best_sum_weight, prune
        if current_sum + suffix_sum[idx] <= best_sum_weight:
            return
            
        # If we've considered all items
        if idx == n:
            if current_sum > best_sum_weight and current_sum < L:
                update_best(current_sum, current_indices)
            return
            
        # Option 1: Take item idx
        # Only valid if current_sum + weights[idx] < L
        if current_sum + weights[idx] < L:
            dfs(idx + 1, current_sum + weights[idx], current_indices + [idx])
            
            # If we timed out, return immediately
            if time.time() - start_time > timeout:
                return
                
        # Option 2: Skip item idx
        dfs(idx + 1, current_sum, current_indices)
        
    # Start DFS
    dfs(0, 0, [])
    
    if not best_indices:
        return [0] # Fallback: 1/2
        
    return best_indices
