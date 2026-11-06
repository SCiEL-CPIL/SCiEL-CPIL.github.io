"""Rewritten program for the suite task; optimized for speed and search depth."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    # Theoretical upper bound from Erdos-Rado Sunflower Lemma
    # Max size without 3-sunflower is at most k!(k-1)^2 = 24 * 9 = 216
    # Actually, the lemma says if size > k!(k-1)^r + 1, sunflower exists.
    # So max size is k!(k-1)^r.
    upper_bound = 216
    
    best_family = []
    start_time = time.time()
    time_limit = 85.0  # Keep 5s margin for 90s kill
    
    # We will perform multiple randomized greedy restarts
    while time.time() - start_time < time_limit:
        random.shuffle(all_masks)
        family = []
        
        # Greedy construction
        for s in all_masks:
            # Check if adding s creates a sunflower with any pair in family
            conflict = False
            # Iterate over pairs in current family
            # Optimization: Break early if conflict found
            for i in range(len(family)):
                a = family[i]
                ha_s = a & s
                for j in range(i + 1, len(family)):
                    b = family[j]
                    # Sunflower condition: a & b == a & s == b & s
                    if (a & b) == ha_s and (b & s) == ha_s:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(s)
                
                # If we hit the theoretical max, we can stop this restart
                if len(family) >= upper_bound:
                    break
        
        # Update best family
        if len(family) > len(best_family):
            best_family = family
            
            # If we hit the theoretical max, we can stop all restarts
            if len(best_family) >= upper_bound:
                break
        
        # Optional: If we have enough time, we could do local search, 
        # but greedy with many restarts is often sufficient for this size.
        
    # Convert bitmasks back to sorted lists of integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
