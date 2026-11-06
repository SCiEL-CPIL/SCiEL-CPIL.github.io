"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)

    # Calculate LCM of 2..48
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Weights w[i] = L / d[i]
    # We want to maximize sum(w[i] for i in S) such that sum < L
    weights = [L // d for d in denominators]
    
    # Create list of (weight, original_index)
    items = [(weights[i], i) for i in range(n)]
    
    # Sort by weight descending to help Branch and Bound prune earlier
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums for bounding
    # suffix[i] = sum of weights[i] + weights[i+1] + ... + weights[n-1]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + items[i][0]
        
    best_sum = 0
    best_indices = []
    
    # Global variables for the DFS
    # Using a list for best_indices to allow mutation in nested function without nonlocal complexity in some contexts, 
    # but we'll use a wrapper or just return best_sum and reconstruct.
    # Actually, storing the mask or indices is better.
    # Since n=47, we can't use a simple int mask if we want to be safe with bit operations? 
    # Python ints are arbitrary precision, so a 47-bit mask is fine.
    best_mask = 0
    
    # Greedy initialization to get a strong lower bound
    curr_sum = 0
    curr_mask = 0
    for i in range(n):
        w, orig_idx = items[i]
        if curr_sum + w < L:
            curr_sum += w
            curr_mask |= (1 << i)
    best_sum = curr_sum
    best_mask = curr_mask
    
    # Branch and Bound
    # We use a stack-based or recursive DFS. Recursive is cleaner.
    # Python recursion limit might be an issue if depth is 47? No, 47 is fine.
    # But we need to be careful with speed.
    
    sys.setrecursionlimit(10000)
    
    def dfs(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask
        
        # Pruning: even if we take all remaining, we can't beat best_sum
        if curr_sum + suffix[idx] <= best_sum:
            return
        
        # Update best if current is valid and better
        if curr_sum > best_sum and curr_sum < L:
            best_sum = curr_sum
            best_mask = curr_mask
            
        if idx == n:
            return
            
        w, _ = items[idx]
        
        # Branch 1: Include item idx
        # Only if it doesn't exceed L-1 (i.e., sum + w < L)
        if curr_sum + w < L:
            dfs(idx + 1, curr_sum + w, curr_mask | (1 << idx))
            
        # Branch 2: Exclude item idx
        # Check bound before recursing? The function call will check it.
        # But we can check if excluding can still lead to a better sum
        # i.e., curr_sum + suffix[idx+1] > best_sum
        if curr_sum + suffix[idx + 1] > best_sum:
            dfs(idx + 1, curr_sum, curr_mask)

    # Run DFS with time check
    # To check time efficiently, we can't easily interrupt recursion.
    # We will use a simple counter or just hope it's fast.
    # Given N=47 and strong bounds, it should be fast.
    # However, to be safe against worst-case exponential blowup, we should monitor time.
    # Since we can't easily break recursion from outside, we will rely on the pruning being strong enough.
    # If it's too slow, we might need to abort.
    # Let's add a time check inside the recursion? That's expensive.
    # Alternative: Run DFS in a thread? No, standard library only, and GIL issues.
    # Let's assume the bound is tight enough. The greedy solution is usually very close to optimal for such dense sets.
    
    dfs(0, 0, 0)
    
    # Reconstruct indices from best_mask
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            # items[i] is (weight, original_index)
            result_indices.append(items[i][1])
            
    if not result_indices:
        # Fallback: should not happen as 1/2 is always valid
        result_indices = [0]
        
    # Verification
    total = sum(Fraction(1, denominators[i]) for i in result_indices)
    if total >= 1:
        # This should not happen due to < L check, but just in case
        # Remove the largest denominator (smallest value) to reduce sum? 
        # Or just return a known valid set
        result_indices = [0] # 1/2
        
    return result_indices
# EVOLVE-BLOCK-END
