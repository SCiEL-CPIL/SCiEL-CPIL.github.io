import time
import math

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    # Sort divisors to have a consistent order, though not strictly necessary for greedy
    divisors.sort()
    
    # 2. Precompute masks for each (m, r)
    # mask[m][r] is an integer with bits set at r, r+m, ...
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for i in range(r, base, m):
                mask_val |= (1 << i)
            masks[m][r] = mask_val
            
    # 3. Greedy Set Cover
    # uncovered is a bitset of all 1s
    uncovered = (1 << base) - 1
    remaining_moduli = set(divisors)
    classes = []
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        # Find the modulus and residue that maximizes coverage of uncovered
        for m in remaining_moduli:
            # For this m, find r that maximizes intersection with uncovered
            # We can iterate through all r for this m
            for r in range(m):
                gain = (uncovered & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        # Select the best class
        classes.append([best_m, best_r])
        
        # Update uncovered
        uncovered &= ~masks[best_m][best_r]
        
        # Remove the modulus from consideration
        remaining_moduli.remove(best_m)
        
        # Time check
        if time.time() > deadline:
            break
            
    return classes
