import time
from fractions import Fraction
import bisect
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline for safety (90s limit, leave margin)
    
    denoms = list(range(2, 49))
    N = len(denoms)
    
    # Split into two halves for meet-in-the-middle
    mid = N // 2  # 23
    half1 = denoms[:mid]  # 23 items
    half2 = denoms[mid:]  # 24 items
    
    n1 = len(half1)
    n2 = len(half2)
    
    # Precompute fractions for exact calculation later
    fracs1 = [Fraction(1, d) for d in half1]
    fracs2 = [Fraction(1, d) for d in half2]
    
    # Generate all subset sums for half1 using DP-like generation
    size1 = 1 << n1
    sums1 = [0.0] * size1
    for i in range(n1):
        val = 1.0 / half1[i]
        for mask in range(1 << i):
            sums1[mask | (1 << i)] = sums1[mask] + val
            
    # Generate all subset sums for half2
    size2 = 1 << n2
    sums2 = [0.0] * size2
    for i in range(n2):
        val = 1.0 / half2[i]
        for mask in range(1 << i):
            sums2[mask | (1 << i)] = sums2[mask] + val
            
    # Create sorted list of (sum, mask) for half2
    pairs2 = [(sums2[m], m) for m in range(size2)]
    pairs2.sort(key=lambda x: x[0])
    sorted_vals2 = [p[0] for p in pairs2]
    
    # Also create a list of unique (sum, mask) pairs to handle ties
    # For each unique sum value, keep all masks that achieve it
    # Actually, for binary search we just need the largest value < target
    # But we need to verify exact sum, so we should check all masks with that value?
    # No - if multiple masks give the same float sum, they may give different exact sums.
    # But the float sum is an approximation. Let's just use the first one found.
    # Actually, we should be more careful. Let's keep all pairs and for a given target,
    # find the largest value < target, then check all masks with that value.
    
    # Better approach: for each distinct value in sorted_vals2, store the list of masks
    # But that's complex. Let's just use the sorted list and for the best candidate,
    # verify the exact sum. If it's not valid (sum >= 1), try the next one.
    
    best_sum_exact = Fraction(0)
    best_mask1 = 0
    best_mask2 = 0
    best_deficit = Fraction(1)  # Initial deficit is 1 (empty set)
    
    # Sort half1 masks by float sum descending to find good candidates early
    # This helps find tight solutions quickly
    mask_order1 = sorted(range(size1), key=lambda m: sums1[m], reverse=True)
    
    # Track how many exact checks we've done
    exact_checks = 0
    max_exact_checks = 500000  # Limit to avoid timeout
    
    for m1 in mask_order1:
        if time.time() > deadline:
            break
        
        s1 = sums1[m1]
        if s1 >= 1.0:
            continue
            
        target = 1.0 - s1
        
        # Find the largest value in sorted_vals2 strictly less than target
        idx = bisect.bisect_left(sorted_vals2, target) - 1
        
        if idx < 0:
            continue
            
        # Check multiple candidates around idx to handle floating point issues
        # and to find the best exact sum
        checked = set()
        for check_idx in range(max(0, idx - 10), min(size2, idx + 11)):
            if check_idx in checked:
                continue
            checked.add(check_idx)
            
            s2 = sorted_vals2[check_idx]
            if s2 >= target:
                continue  # Not strictly below
                
            m2 = pairs2[check_idx][1]
            combined_float = s1 + s2
            
            if combined_float >= 1.0:
                continue  # Floating point says it's >= 1, skip
                
            # Calculate exact sum
            s_total = Fraction(0)
            
            temp_mask = m1
            i = 0
            while temp_mask:
                if temp_mask & 1:
                    s_total += fracs1[i]
                temp_mask >>= 1
                i += 1
                
            temp_mask = m2
            i = 0
            while temp_mask:
                if temp_mask & 1:
                    s_total += fracs2[i]
                temp_mask >>= 1
                i += 1
                
            exact_checks += 1
            
            # Check validity
            if s_total < 1:
                deficit = 1 - s_total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_sum_exact = s_total
                    best_mask1 = m1
                    best_mask2 = m2
                    
        if exact_checks > max_exact_checks:
            break
            
    # If no valid sum found (should not happen)
    if best_mask1 == 0 and best_mask2 == 0:
        # Fallback: use just the first element
        return [0]
        
    # Construct indices
    indices = []
    # Indices for half1
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # Indices for half2
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    if not indices:
        return [0]
        
    return indices
