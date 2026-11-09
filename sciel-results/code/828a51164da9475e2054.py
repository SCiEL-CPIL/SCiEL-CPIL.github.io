import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    # Split into two halves
    # 47 items. 23 and 24.
    half1 = n // 2
    half2 = n - half1
    
    w1 = weights[:half1]
    w2 = weights[half1:]
    
    # Generate sums for half 1
    n1 = len(w1)
    size1 = 1 << n1
    sums1 = [0] * size1
    for i in range(n1):
        bit = 1 << i
        w = w1[i]
        for mask in range(bit):
            sums1[bit | mask] = sums1[mask] + w
            
    # Generate sums for half 2
    n2 = len(w2)
    size2 = 1 << n2
    sums2 = [0] * size2
    for i in range(n2):
        bit = 1 << i
        w = w2[i]
        for mask in range(bit):
            sums2[bit | mask] = sums2[mask] + w
            
    # Prepare half 2 for binary search
    # We need to map sum -> mask? No, we just need to find the best sum.
    # But we need the mask to reconstruct.
    # Since multiple masks can produce the same sum, we can just keep one mask per sum?
    # No, we just need ANY mask that produces the best total sum.
    # So we can store (sum, mask) pairs, sort by sum.
    
    pairs2 = [(sums2[mask], mask) for mask in range(size2)]
    pairs2.sort()
    
    # Extract just the sums for bisect
    sums2_sorted = [p[0] for p in pairs2]
    
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through half 1 sums
    for mask1 in range(size1):
        if time.time() > deadline:
            break
            
        s1 = sums1[mask1]
        
        # We need s1 + s2 < L
        # s2 < L - s1
        limit = L - s1
        
        # Find largest s2 < limit
        # bisect_left finds insertion point for limit
        # idx is index of first element >= limit
        # So idx - 1 is largest element < limit
        idx = bisect.bisect_left(sums2_sorted, limit)
        
        if idx > 0:
            s2 = sums2_sorted[idx - 1]
            total = s1 + s2
            deficit = L - total
            
            # Check if this is a valid non-empty set
            # If mask1 == 0 and mask2 == 0, sum is 0, deficit is L.
            # We want to minimize deficit, so we prefer smaller deficit.
            # Empty set has deficit L. Any other set has deficit < L (since sum > 0).
            # So if we find any valid set, it will beat the empty set.
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = mask1
                best_mask2 = pairs2[idx - 1][1]
                
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            result_indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            result_indices.append(half1 + i)
            
    # Ensure non-empty
    if not result_indices:
        # Fallback: pick index 0 (denominator 2)
        result_indices = [0]
        
    return result_indices
