import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15 corresponding to 1..16
    all_subsets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_subsets.append(mask)
    
    best_family_size = 0
    best_family = []
    
    # Number of restarts depends on time available
    # Each restart takes roughly O(N^3) ops. 
    # If N=100, N^3=1e6. 1820 * 1e6 is too much for one restart if we check all pairs for each candidate?
    # Wait, the check for one candidate S against family of size K is O(K^2).
    # Total for one restart: sum_{k=0}^{M} k^2 * (cost of intersection check).
    # Cost of intersection check is constant (bit ops).
    # If M=200, sum k^2 ~ 2.6e6.
    # 1820 candidates * 2.6e6 checks? No, the sum is over the growth of the family.
    # The total number of pair checks is sum_{k=1}^{M} k^2 * (number of candidates considered).
    # Actually, for each candidate, we check against the current family.
    # If we process 1820 candidates, and family grows to M, the total work is roughly 1820 * M^2 / 2?
    # No, the family size increases.
    # Let's just run it and see.
    
    num_restarts = 20 # Start with a reasonable number
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0: # Stop if close to deadline
            break
            
        random.shuffle(all_subsets)
        family = []
        
        for s in all_subsets:
            # Check if s can be added
            valid = True
            # Iterate over all pairs in family
            # Optimization: if len(family) is large, this is the bottleneck.
            for i in range(len(family)):
                a = family[i]
                # Optimization: precompute intersections? No, s changes.
                # But we can break early.
                for j in range(i + 1, len(family)):
                    b = family[j]
                    core_ab = a & b
                    if (a & s) == core_ab and (b & s) == core_ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
                
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
        # If we hit a high quality, maybe reduce restarts?
        # Or just let it run.
        
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(16):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
