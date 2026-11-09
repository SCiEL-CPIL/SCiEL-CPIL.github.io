import time
import numpy as np

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    L = 1
    L *= 16
    L *= 27
    L *= 25
    L *= 7
    L *= 11
    L *= 13
    L *= 17
    L *= 19
    L *= 23
    L *= 29
    L *= 31
    L *= 37
    L *= 41
    L *= 43
    L *= 47
    
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    target = L - 1
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    n1 = len(w1)
    n2 = len(w2)
    
    # Generate subset sums for w1
    # n1 is 23
    # 2^23 = 8388608
    masks1 = np.arange(2**n1, dtype=np.int64)
    # Create bit matrix
    # (2**n1, n1)
    # This might be memory heavy if not careful, but we delete it immediately
    bits1 = (masks1[:, None] >> np.arange(n1)) & 1
    sums1 = bits1 @ w1
    del bits1
    
    # Filter sums1 <= target
    valid1 = sums1 <= target
    sums1 = sums1[valid1]
    masks1 = masks1[valid1]
    
    # Generate subset sums for w2
    # n2 is 24
    # 2^24 = 16777216
    masks2 = np.arange(2**n2, dtype=np.int64)
    bits2 = (masks2[:, None] >> np.arange(n2)) & 1
    sums2 = bits2 @ w2
    del bits2
    
    # Filter sums2 <= target
    valid2 = sums2 <= target
    sums2 = sums2[valid2]
    masks2 = masks2[valid2]
    
    # Sort sums2 and masks2 together
    # We need to sort sums2 and keep track of which mask corresponds to which sum
    order = np.argsort(sums2)
    sorted_sums2 = sums2[order]
    sorted_masks2 = masks2[order]
    
    # For each sum in sums1, find the largest sum in sorted_sums2 such that sum1 + sum2 <= target
    # limits = target - sums1
    limits = target - sums1
    
    # searchsorted returns index where limit would be inserted to maintain order
    # side='right' means we want the index of the first element > limit
    # So idx - 1 is the largest element <= limit
    idxs = np.searchsorted(sorted_sums2, limits, side='right') - 1
    
    # Check for valid indices (idx >= 0)
    valid_idxs = idxs >= 0
    
    if not np.any(valid_idxs):
        # Should not happen as 0 is always a valid sum
        return [0] # Fallback, though 0 sum is valid
    
    # Calculate candidate total sums
    # sums1[valid_idxs] + sorted_sums2[idxs[valid_idxs]]
    cands = sums1[valid_idxs] + sorted_sums2[idxs[valid_idxs]]
    
    # Find the maximum
    best_local_idx = np.argmax(cands)
    best_sum = cands[best_local_idx]
    
    # Recover masks
    # The index in the filtered arrays
    # valid_idxs is a boolean mask over the original sums1 array
    # We need the index in the original sums1 array corresponding to best_local_idx
    # This is tricky. valid_idxs is a boolean array.
    # Let's create an array of indices where valid_idxs is True
    valid_indices = np.where(valid_idxs)[0]
    
    orig_idx1 = valid_indices[best_local_idx]
    mask1 = masks1[orig_idx1]
    
    orig_idx2_in_sorted = idxs[valid_idxs[best_local_idx]]
    mask2 = sorted_masks2[orig_idx2_in_sorted]
    
    # Reconstruct indices
    indices = []
    
    # Decode mask1
    m = mask1
    for i in range(n1):
        if m & (1 << i):
            indices.append(i)
            
    # Decode mask2
    m = mask2
    for i in range(n2):
        if m & (1 << i):
            indices.append(n1 + i)
            
    return indices
