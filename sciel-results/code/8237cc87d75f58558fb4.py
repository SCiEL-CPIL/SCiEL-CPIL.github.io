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
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            for mask in range(bit):
                sums[bit | mask] = sums[mask] + w
        
        valid_pairs = [(sums[mask], mask) for mask in range(total_masks) if sums[mask] < L]
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    # Sort left by sum DESCENDING to check best candidates first
    left_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Sort right by sum ASCENDING for two-pointer
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted right sums and masks
    right_sums = [p[0] for p in right_pairs]
    right_masks = [p[1] for p in right_pairs]
    
    if time.time() > deadline:
        return [0]
    
    # Two-pointer search
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    # j points to the largest valid index in right_sums
    j = -1
    n_right = len(right_sums)
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        target = L - ls
        
        # Move j forward while the next right sum is still < target
        # Since ls is decreasing, target is increasing, so j only moves forward
        while j + 1 < n_right and right_sums[j + 1] < target:
            j += 1
            
        if j >= 0:
            rs = right_sums[j]
            rm = right_masks[j]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lm
                best_right_mask = rm
                if deficit == 1:
                    break
    
    # Fallback
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
