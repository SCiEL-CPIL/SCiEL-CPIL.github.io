"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    # All 4-element subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)
    
    # Precompute pairwise intersections as a lookup? No, too large.
    # Instead, use a greedy approach with efficient checking.
    
    best_family = []
    
    # Try different orderings to find a larger family
    # Strategy: greedy with bitmask operations
    
    import random
    random.seed(42)
    
    def check_violation(family, new_mask):
        """Check if adding new_mask creates a sunflower violation."""
        n = len(family)
        for i in range(n):
            a = family[i]
            ab = a & new_mask
            for j in range(i + 1, n):
                b = family[j]
                # Check if A∩B = A∩C = B∩C where C = new_mask
                # A∩B = ab (already computed)
                # A∩C = ab (same)
                # B∩C = b & new_mask
                bc = b & new_mask
                if ab == bc:
                    return True
        return False
    
    # Greedy: try to add sets in a good order
    # First pass: natural order
    family = []
    for mask in all_sets:
        if time.time() > deadline:
            break
        if not check_violation(family, mask):
            family.append(mask)
    
    best_family = family[:]
    
    # Second pass: try random shuffles to potentially find larger families
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        shuffled = all_sets[:]
        random.shuffle(shuffled)
        
        family = []
        for mask in shuffled:
            if time.time() > deadline:
                break
            if not check_violation(family, mask):
                family.append(mask)
        
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Convert back to sorted lists of 1..16
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
