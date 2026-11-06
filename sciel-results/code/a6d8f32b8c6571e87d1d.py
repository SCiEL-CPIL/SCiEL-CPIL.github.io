import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_family = []
    start_time = time.time()
    max_time = 50.0 # Leave buffer for overhead
    
    while time.time() - start_time < max_time:
        random.shuffle(all_sets)
        family = []
        pair_intersections = [] # List of (a_mask, b_mask, inter_mask)
        
        for s in all_sets:
            valid = True
            # Check against all existing pairs
            for a, b, i in pair_intersections:
                if (a & s == i) and (b & s == i):
                    valid = False
                    break
            
            if valid:
                family.append(s)
                # Update pair_intersections with new pairs involving s
                for a in family[:-1]: # All previous elements
                    i = a & s
                    pair_intersections.append((a, s, i))
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
