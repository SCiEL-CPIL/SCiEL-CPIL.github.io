"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    def can_add(mask_new, chosen_masks):
        """Check if mask_new can be added without forming a sunflower."""
        L = len(chosen_masks)
        if L < 2:
            return True
        
        # Group indices by their intersection with mask_new
        groups = {}
        for i in range(L):
            inter = chosen_masks[i] & mask_new
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(i)
        
        # For each group with 2+ members, check if any pair has mutual intersection == inter
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            for ii in range(len(indices)):
                i = indices[ii]
                mi = chosen_masks[i]
                for jj in range(ii + 1, len(indices)):
                    j = indices[jj]
                    mj = chosen_masks[j]
                    if (mi & mj) == inter:
                        return False
        return True
    
    best_family = []
    best_size = 0
    
    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add(mask, chosen):
            chosen.append(mask)
            
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy with many restarts
    rng = random.Random(42)
    
    attempt = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
        
        attempt += 1
        rng.seed(12345 + attempt * 7919)
        shuffled = all_subsets[:]
        rng.shuffle(shuffled)
        
        chosen = []
        
        for mask in shuffled:
            if time.time() > deadline - 2.0:
                break
            if can_add(mask, chosen):
                chosen.append(mask)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            if best_size >= 250:
                break
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
