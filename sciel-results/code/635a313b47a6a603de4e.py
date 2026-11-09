import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

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
        total_masks = 1 << size
        sums = [0] * total_masks
        sums_local = sums
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums_local[bit | mask] = sums_local[mask] + w
        
        valid_pairs = [(sums[mask], mask) for mask in range(total_masks) if sums[mask] < L]
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    # Sort by sum
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted right sums for binary search
    right_sums = [p[0] for p in right_pairs]
    right_masks = [p[1] for p in right_pairs]
    
    if time.time() > deadline:
        return [0]
    
    # For each left sum, binary search for the best right sum
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        # We need ls + rs < L, so rs < L - ls
        # Find the largest right sum < (L - ls)
        target = L - ls
        # bisect_left finds the first index where right_sums[i] >= target
        # We want the last index where right_sums[i] < target
        idx = bisect.bisect_left(right_sums, target) - 1
        
        if idx >= 0:
            rs = right_sums[idx]
            rm = right_masks[idx]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
                if deficit == 1:
                    break
    
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
