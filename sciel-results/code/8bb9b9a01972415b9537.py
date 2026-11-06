"""Combined program: Optimized Meet-in-the-Middle exact search for optimal subset sum."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    n1 = mid
    n2 = n - mid
    
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    size1 = 1 << n1
    size2 = 1 << n2
    
    # Generate subset sums for first half
    sums1 = [0] * size1
    masks1 = [0] * size1
    for i in range(n1):
        w = w1[i]
        for j in range(1 << i):
            sums1[(1 << i) | j] = sums1[j] + w
            masks1[(1 << i) | j] = masks1[j] | (1 << i)
            
    # Generate subset sums for second half
    sums2 = [0] * size2
    masks2 = [0] * size2
    for i in range(n2):
        w = w2[i]
        for j in range(1 << i):
            sums2[(1 << i) | j] = sums2[j] + w
            masks2[(1 << i) | j] = masks2[j] | (1 << i)
            
    # Filter and sort second half
    # We only care about sums < L
    # Create list of (sum, mask)
    pairs2 = []
    for mask in range(size2):
        s = sums2[mask]
        if s < L:
            pairs2.append((s, mask))
            
    # Sort by sum
    pairs2.sort(key=lambda x: x[0])
    
    # Sort first half
    # We need to iterate through them in sorted order for two-pointer
    pairs1 = []
    for mask in range(size1):
        s = sums1[mask]
        if s < L: # Optimization: s1 must be < L to have any chance (since s2 >= 0)
            pairs1.append((s, mask))
            
    pairs1.sort(key=lambda x: x[0])
    
    # Two-pointer search
    # i points to smallest sum in pairs1
    # j points to largest sum in pairs2
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    i = 0
    j = len(pairs2) - 1
    
    # We want max s1 + s2 < L
    # If s1 + s2 >= L, decrease j (smaller s2)
    # If s1 + s2 < L, it's a candidate. Increase i (larger s1) to try to get closer to L
    
    while i < len(pairs1) and j >= 0:
        s1, m1 = pairs1[i]
        s2, m2 = pairs2[j]
        
        total = s1 + s2
        
        if total >= L:
            j -= 1
        else:
            # total < L
            if total > best_sum:
                best_sum = total
                best_mask1 = m1
                best_mask2 = m2
                
            i += 1
            
    # Check time
    if time.time() - start_time > timeout:
        # Fallback: return a known good heuristic solution
        # Greedy: pick smallest denominators first? No, pick largest first to fill?
        # Actually, just return [0] (1/2) which is valid.
        return [0]

    # Reconstruct indices
    indices = []
    for i_idx in range(n1):
        if best_mask1 & (1 << i_idx):
            indices.append(i_idx)
            
    for i_idx in range(n2):
        if best_mask2 & (1 << i_idx):
            indices.append(mid + i_idx)
            
    if not indices:
        indices = [0]
        
    return indices
# EVOLVE-BLOCK-END
