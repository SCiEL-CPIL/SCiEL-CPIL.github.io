import time
from fractions import Fraction
import bisect

def construct():
    deadline = time.time() + 55  # 55s deadline for safety
    
    denoms = list(range(2, 49))
    N = len(denoms)
    
    # Split into two halves
    mid = N // 2  # 23
    half1 = denoms[:mid]  # 23 items
    half2 = denoms[mid:]  # 24 items
    
    n1 = len(half1)
    n2 = len(half2)
    
    # Precompute fractions for exact calculation later
    fracs1 = [Fraction(1, d) for d in half1]
    fracs2 = [Fraction(1, d) for d in half2]
    
    # Generate all subset sums for half1
    # Store (float_sum, mask)
    sums1 = []
    for mask in range(1 << n1):
        s = 0.0
        # Optimization: iterate bits
        # m = mask
        # while m:
        #   lsb = m & -m
        #   idx = (lsb.bit_length() - 1)
        #   s += 1.0 / half1[idx]
        #   m -= lsb
        # But simple loop is fine for 23 bits if optimized
        # Let's use a recursive or iterative generation to be faster?
        # Actually, 2^23 is 8 million. A simple loop with bit checks might be slow in Python.
        # Let's use a DP-like generation or just iterate.
        pass
        
    # Faster generation:
    # sums1 = [0.0] * (1 << n1)
    # for i in range(n1):
    #   val = 1.0 / half1[i]
    #   for mask in range(1 << i):
    #     sums1[mask | (1 << i)] = sums1[mask] + val
    # This is efficient.
    
    sums1 = [0.0] * (1 << n1)
    for i in range(n1):
        val = 1.0 / half1[i]
        # Iterate over masks that don't have bit i set
        # Range 0 to 2^i - 1 covers the lower bits
        for mask in range(1 << i):
            sums1[mask | (1 << i)] = sums1[mask] + val
            
    # Generate all subset sums for half2
    sums2 = [0.0] * (1 << n2)
    for i in range(n2):
        val = 1.0 / half2[i]
        for mask in range(1 << i):
            sums2[mask | (1 << i)] = sums2[mask] + val
            
    # We need to pair sums1 and sums2.
    # For each sum in sums1, we want the largest sum in sums2 such that sum1 + sum2 < 1.
    # This is equivalent to sum2 < 1 - sum1.
    
    # Create a list of (sum, mask) for half2 and sort it
    # To save memory/time, we can just sort the values and keep track of indices?
    # We need the mask to reconstruct the indices.
    # List of tuples (float_val, mask)
    pairs2 = [(sums2[m], m) for m in range(1 << n2)]
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted values for binary search
    sorted_vals2 = [p[0] for p in pairs2]
    
    best_sum_exact = Fraction(0)
    best_mask1 = 0
    best_mask2 = 0
    best_float_sum = 0.0
    
    # Sort half1 masks by float sum descending to find good candidates early
    # This sets a tight best_float_sum threshold quickly, reducing exact computations
    mask_order1 = sorted(range(1 << n1), key=lambda m: sums1[m], reverse=True)
    
    for m1 in mask_order1:
        s1 = sums1[m1]
        if s1 >= 1.0:
            continue
            
        target = 1.0 - s1
        
        # Find the largest value in sorted_vals2 strictly less than target
        # bisect_left returns the first index where val >= target
        # We want the last index where val < target
        # So idx = bisect.bisect_left(sorted_vals2, target) - 1
        idx = bisect.bisect_left(sorted_vals2, target) - 1
        
        if idx < 0:
            continue
            
        s2 = sorted_vals2[idx]
        combined_float = s1 + s2
        
        # Only check exact if it's potentially better
        # We can maintain a threshold
        if combined_float > best_float_sum:
            # Get the mask for s2
            m2 = pairs2[idx][1]
            
            # Calculate exact sum
            # Sum of fractions for mask1
            s1_exact = Fraction(0)
            # Optimization: Precompute exact sums for half1? 
            # 8 million Fractions is too much memory/time to precompute all.
            # But we only do this for improvements.
            # How many improvements will there be?
            # The float sum increases monotonically as we find better candidates?
            # No, we iterate m1 in arbitrary order.
            # We can sort m1 by s1 descending? 
            # If we process s1 from largest to smallest, the target 1-s1 becomes smaller.
            # The best s2 will also become smaller.
            # The combined sum might not be monotonic.
            
            # Let's just compute exact sum.
            # To speed up, we can precompute exact sums for half1 and half2?
            # 2^23 Fractions ~ 8M objects. Memory ~ 8M * 50 bytes = 400MB. Risky.
            # Let's compute on the fly.
            
            # Compute exact total sum
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
                
            # Check validity
            if s_total < 1:
                if s_total > best_sum_exact:
                    best_sum_exact = s_total
                    best_mask1 = m1
                    best_mask2 = m2
                    best_float_sum = combined_float
                    
            if time.time() > deadline:
                break
                
    # If no valid sum found (should not happen, empty set sum=0 is valid)
    if best_sum_exact == Fraction(0) and best_mask1 == 0 and best_mask2 == 0:
        # Check if empty set was considered.
        # m1=0, m2=0 -> sum=0.
        # bisect_left(1.0) - 1.
        # If there are no sums < 1.0, idx < 0.
        # But 0.0 is a sum. 0.0 < 1.0.
        # So it should find it.
        pass
        
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
        return [0] # Fallback, though 0 corresponds to denom 2, sum 0.5 < 1.
        
    return indices
