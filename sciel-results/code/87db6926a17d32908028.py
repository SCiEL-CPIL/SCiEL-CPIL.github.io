"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    # All 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_masks.append(mask)
    
    def is_sunflower(a, b, c):
        """Check if three 4-sets form a sunflower (all pairwise intersections equal)."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def can_add(family, mask):
        """Check if mask can be added to family without creating a sunflower."""
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_sunflower(family[i], family[j], mask):
                    return False
        return True
    
    best_family = []
    
    # Multiple random greedy passes
    for trial in range(50):
        shuffled = all_masks[:]
        random.shuffle(shuffled)
        family = []
        for mask in shuffled:
            if can_add(family, mask):
                family.append(mask)
        
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Local search on best: try to improve by swapping
    family = best_family[:]
    improved = True
    while improved:
        improved = False
        # Try adding any missing set
        for mask in all_masks:
            if mask not in family and can_add(family, mask):
                family.append(mask)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-2 swaps: remove one, add two
        for remove_idx in range(len(family)):
            removed = family[remove_idx]
            family.pop(remove_idx)
            
            added = []
            for mask in all_masks:
                if mask == removed or mask in family:
                    continue
                if can_add(family, mask):
                    added.append(mask)
                    if len(added) == 2:
                        break
            
            if len(added) > 1:
                family.extend(added)
                improved = True
            else:
                family.append(removed)
            if improved:
                break
    
    # Convert to sorted lists
    result = []
    for mask in family:
        elems = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
