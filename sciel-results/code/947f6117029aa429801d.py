import sys
from math import gcd
from bisect import bisect_right

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, ..., 48)
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Compute weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    n1 = len(w1)
    n2 = len(w2)
    
    # Generate subset sums for first half
    sums1 = [0] * (1 << n1)
    for i in range(n1):
        w = w1[i]
        for j in range(1 << i):
            sums1[(1 << i) | j] = sums1[j] + w
            
    # Generate subset sums for second half
    sums2 = [0] * (1 << n2)
    for i in range(n2):
        w = w2[i]
        for j in range(1 << i):
            sums2[(1 << i) | j] = sums2[j] + w
            
    # Filter and prepare second half for binary search
    # We need (sum, mask) pairs where sum < L
    
    # Create list of indices for valid sums in second half
    valid_indices2 = []
    for mask in range(1 << n2):
        if sums2[mask] < L:
            valid_indices2.append(mask)
            
    # Sort valid_indices2 based on sums2 values
    # Using sorted with key
    valid_indices2.sort(key=lambda m: sums2[m])
    
    # Extract sorted sums and masks
    sorted_sums2 = [sums2[m] for m in valid_indices2]
    sorted_masks2 = valid_indices2 # The masks are already sorted by sum
    
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over first half
    for mask1 in range(1 << n1):
        s1 = sums1[mask1]
        if s1 >= L:
            continue
            
        target = L - 1 - s1
        
        # Find largest sum in second half <= target
        # bisect_right gives index of first element > target
        # So index - 1 is the largest <= target
        idx = bisect_right(sorted_sums2, target)
        
        if idx > 0:
            # The best candidate in second half is at idx - 1
            s2 = sorted_sums2[idx - 1]
            m2 = sorted_masks2[idx - 1]
            total_sum = s1 + s2
            
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = mask1
                best_mask2 = m2
                
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        indices = [0]
        
    return indices
