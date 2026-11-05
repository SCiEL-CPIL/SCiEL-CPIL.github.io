import time
from math import gcd
from functools import reduce
import bisect
from fractions import Fraction

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a meet-in-the-middle approach with exact integer arithmetic and pruning.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators to work with integers
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Values in integer scale: V[i] = L // denominators[i]
    V = [L // d for d in denominators]
    
    # Split into two halves for meet-in-the-middle
    # n=47. Let's split 23 and 24.
    n1 = 23
    n2 = n - n1
    V1 = V[:n1]
    V2 = V[n1:]
    
    def gen_sums(vals):
        sums = [0]
        masks = [0]
        for i, v in enumerate(vals):
            if time.time() > deadline:
                return sums, masks
            
            # Pruning: only keep sums that are strictly less than L
            # If s + v >= L, this subset cannot be part of a valid solution 
            # because all other values are positive.
            new_sums = []
            new_masks = []
            for s, m in zip(sums, masks):
                ns = s + v
                if ns < L:
                    new_sums.append(ns)
                    new_masks.append(m | (1 << i))
            
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks

    # Generate V1 sums
    sums1, masks1 = gen_sums(V1)
    
    if time.time() > deadline:
        return [0] 

    # Generate V2 sums
    sums2, masks2 = gen_sums(V2)
    
    if time.time() > deadline:
        return [0]

    # Sort sums2 and masks2 together by sum
    # We only need the best (largest) sum for each mask? No, different masks can have same sum.
    # We just need the list of (sum, mask) sorted by sum.
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_total = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through sums1
    # Optimization: If sums1 is large, we can also sort it and use two pointers?
    # But bisect is O(log N) per query.
    for i in range(len(sums1)):
        s1 = sums1[i]
        # s1 is already < L due to pruning
        
        # We want s2 such that s1 + s2 < L
        # s2 < L - s1
        # s2 <= L - 1 - s1
        target = L - 1 - s1
        
        # Find rightmost index in sorted_sums2 <= target
        idx = bisect.bisect_right(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            m2 = pairs2[idx - 1][1]
            
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask1 = masks1[i]
                best_mask2 = m2
                
        if time.time() > deadline:
            break

    # Reconstruct indices
    indices = []
    for j in range(n1):
        if best_mask1 & (1 << j):
            indices.append(j)
            
    for j in range(n2):
        if best_mask2 & (1 << j):
            indices.append(n1 + j)
            
    if not indices:
        return [0]

    # Verify result
    current_sum_frac = Fraction(0, 1)
    for idx in indices:
        current_sum_frac += Fraction(1, denominators[idx])
        
    if current_sum_frac >= 1:
        # Should not happen, but fallback
        return [0]

    return indices

if __name__ == "__main__":
    res = construct()
    pass
