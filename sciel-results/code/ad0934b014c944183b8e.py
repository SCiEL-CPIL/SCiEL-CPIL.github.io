import math
import numpy as np

def construct():
    # Calculate LCM of 2 to 48
    lcm_val = 1
    for i in range(2, 49):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
    
    # Denominators are 2 to 48
    denoms = list(range(2, 49))
    weights = [lcm_val // d for d in denoms]
    
    n = len(weights) # 47
    n1 = 23
    n2 = n - n1 # 24
    
    W1 = weights[:n1]
    W2 = weights[n1:]
    
    def gen_sums_masks(weights_list):
        size = 1 << len(weights_list)
        sums = np.zeros(size, dtype=np.int64)
        masks = np.zeros(size, dtype=np.uint64)
        current_size = 1
        for i, w in enumerate(weights_list):
            sums[current_size:current_size*2] = sums[:current_size] + w
            masks[current_size:current_size*2] = masks[:current_size] | (1 << i)
            current_size *= 2
        return sums, masks

    sums1, masks1 = gen_sums_masks(W1)
    sums2, masks2 = gen_sums_masks(W2)
    
    # Sort sums2 and keep masks aligned
    order = np.argsort(sums2)
    sums2_sorted = sums2[order]
    masks2_sorted = masks2[order]
    
    # We want s1 + s2 < L
    # s2 < L - s1
    # s2 <= L - s1 - 1
    targets = lcm_val - sums1 - 1
    
    # Find index of largest element in sums2_sorted <= target
    indices = np.searchsorted(sums2_sorted, targets, side='right') - 1
    
    # Filter out invalid indices
    valid = indices >= 0
    
    if not np.any(valid):
        # Fallback to empty set (should not happen)
        return []
    
    valid_indices = indices[valid]
    valid_sums1 = sums1[valid]
    valid_sums2 = sums2_sorted[valid_indices]
    valid_masks1 = masks1[valid]
    valid_masks2 = masks2_sorted[valid_indices]
    
    total_sums = valid_sums1 + valid_sums2
    max_idx = np.argmax(total_sums)
    
    best_mask1 = valid_masks1[max_idx]
    best_mask2 = valid_masks2[max_idx]
    
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            result_indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            result_indices.append(n1 + i)
            
    return result_indices
