import time
import bisect
from math import lcm

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Function to generate all subset sums and masks
    def generate_sums(weights_list):
        size = len(weights_list)
        sums = [0] * (1 << size)
        # Standard DP to generate subset sums
        # sums[mask] = sums[mask without lowest set bit] + weight[lowest set bit]
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + w
        
        # Filter out sums >= L and keep (sum, mask) pairs
        valid_pairs = []
        for mask in range(1 << size):
            s = sums[mask]
            if s < L:
                valid_pairs.append((s, mask))
        
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    right_pairs = generate_sums(right_weights)
    
    # Sort by sum
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract just the sums for binary search
    right_sums = [rs for rs, _ in right_pairs]
    
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    # For each left sum, find the largest right sum such that left_sum + right_sum < L
    for ls, lm in left_pairs:
        # We need: rs < L - ls
        # So we want the largest rs < L - ls
        target = L - ls
        # Find the rightmost index where right_sums[index] < target
        # bisect_left gives first position >= target, so we want that - 1
        idx = bisect.bisect_left(right_sums, target) - 1
        
        if idx >= 0:
            rs, rm = right_pairs[idx]
            total = ls + rs
            if total < L:
                deficit = L - total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_mask = lm
                    best_right_mask = rm
    
    # Fallback if no valid pair found
    if best_left_mask == -1:
        best_left_mask = 0
        best_right_mask = 0
        best_deficit = L

    # Reconstruct indices
    result_indices = []
    for k in range(left_size):
        if best_left_mask & (1 << k):
            result_indices.append(k)
            
    for k in range(right_size):
        if best_right_mask & (1 << k):
            result_indices.append(left_size + k)
            
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
        
    return result_indices
