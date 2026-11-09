import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin

    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Weights are L / d
    weights = [L // d for d in denoms]
    
    # Split into two halves
    # 23 and 24
    half = n // 2
    left_size = half
    right_size = n - half
    
    w_left = weights[:left_size]
    w_right = weights[left_size:]
    
    # Generate subsets for a list of weights
    # Returns lists of sums and masks
    def generate_subsets(weights_part, limit):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_part):
            # Current length
            current_len = len(sums)
            bit = 1 << i
            # List comprehension for new sums
            # We only add if s + w < limit
            # Since we process weights in any order, this is valid.
            # Optimization: If w is large, few will fit.
            # Using list comprehension is faster than a for loop with append.
            new_sums = []
            new_masks = []
            # Local variables for speed
            s_list = sums
            m_list = masks
            lim = limit
            ww = w
            bb = bit
            for j in range(current_len):
                ns = s_list[j] + ww
                if ns < lim:
                    new_sums.append(ns)
                    new_masks.append(m_list[j] | bb)
            
            if new_sums:
                sums.extend(new_sums)
                masks.extend(new_masks)
                
        return sums, masks

    # Generate right subsets first as it's larger
    # We need to sort them, so we need them in memory.
    # Converting to numpy arrays saves memory in the combination step if we use them there,
    # but the generation itself uses Python lists.
    
    right_sums_list, right_masks_list = generate_subsets(w_right, L)
    
    # Convert to numpy arrays
    # Use int64 to be safe, though L fits in int64.
    # Masks also fit in int64 (24 bits).
    r_sums = np.array(right_sums_list, dtype=np.int64)
    r_masks = np.array(right_masks_list, dtype=np.int64)
    
    # Sort right sums and reorder masks
    # np.argsort is stable? We don't care about stability, just order.
    order = np.argsort(r_sums)
    r_sums_sorted = r_sums[order]
    r_masks_sorted = r_masks[order]
    
    # Free the unsorted arrays to save memory
    del r_sums
    del r_masks
    del right_sums_list
    del right_masks_list
    import gc
    gc.collect()

    # Generate left subsets
    left_sums_list, left_masks_list = generate_subsets(w_left, L)
    
    # Convert to numpy arrays
    l_sums = np.array(left_sums_list, dtype=np.int64)
    l_masks = np.array(left_masks_list, dtype=np.int64)
    
    # We want to find max l_sum + r_sum < L
    # For each l_sum, find max r_sum < L - l_sum
    # limits = L - l_sums
    limits = L - l_sums
    
    # Use searchsorted to find the index where limit would be inserted
    # side='left' gives the first index where r_sums_sorted[idx] >= limit
    # We want the largest r_sum strictly less than limit, so idx - 1
    # If idx == 0, there is no such r_sum (r_sum >= 0 always, so if limit <= 0, no match)
    # But l_sum < L, so limit > 0.
    # Also r_sums contains 0. So idx >= 1 always if limit > 0.
    
    indices_right = np.searchsorted(r_sums_sorted, limits, side='left') - 1
    
    # Filter out invalid indices (though should be none if limits > 0)
    valid_mask = indices_right >= 0
    
    if not np.any(valid_mask):
        # Fallback, should not happen
        return [n - 1]
    
    l_sums_valid = l_sums[valid_mask]
    l_masks_valid = l_masks[valid_mask]
    r_indices_valid = indices_right[valid_mask]
    
    r_sums_valid = r_sums_sorted[r_indices_valid]
    r_masks_valid = r_masks_sorted[r_indices_valid]
    
    total_sums = l_sums_valid + r_sums_valid
    
    # Find the index of the maximum total sum
    best_idx = np.argmax(total_sums)
    
    best_l_sum = l_sums_valid[best_idx]
    best_r_sum = r_sums_valid[best_idx]
    best_l_mask = l_masks_valid[best_idx]
    best_r_mask = r_masks_valid[best_idx]
    
    # Decode masks to indices
    indices = []
    
    # Left half corresponds to 0..left_size-1
    for i in range(left_size):
        if best_l_mask & (1 << i):
            indices.append(i)
            
    # Right half corresponds to left_size..n-1
    for i in range(right_size):
        if best_r_mask & (1 << i):
            indices.append(left_size + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback: pick the smallest fraction (largest denominator)
        indices = [n - 1]
        
    return indices
