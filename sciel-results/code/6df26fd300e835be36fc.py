"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Use bitmask representation for speed
    mask_map = {i: 1 << (i-1) for i in range(1, 17)}
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= mask_map[x]
        return m
        
    masks = [to_mask(s) for s in all_sets]
    mask_to_set = {to_mask(s): s for s in all_sets}
    
    # Precompute pairwise intersections for all pairs of masks?
    # That would be C(1820, 2) ~ 1.6M entries, each an int. Might be too much memory/time.
    # Instead, we'll compute on the fly but use efficient checks.
    
    def is_valid_addition(cand_mask, family):
        """
        Check if cand_mask can be added to family without creating a sunflower.
        A sunflower is 3 distinct sets A, B, C where A∩B = A∩C = B∩C.
        For each pair (A, B) in family, check if {A, B, cand} forms a sunflower.
        """
        n = len(family)
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                ab = a & b
                ac = a & cand_mask
                bc = b & cand_mask
                if ab == ac == bc:
                    return False
        return True
    
    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy: Multiple greedy passes with different orderings, then local improvement
    num_greedy_trials = 20
    
    for trial in range(num_greedy_trials):
        if time.time() > deadline - 10:  # Leave time for local search
            break
        
        if trial == 0:
            # Natural order
            order = list(range(n_sets))
        else:
            # Random shuffle
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family = []
        family_set = set()
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            cand = masks[idx]
            
            # Skip if already in family
            if cand in family_set:
                continue
            
            if is_valid_addition(cand, family):
                family.append(cand)
                family_set.add(cand)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Early stop if we hit the theoretical max
        if best_size >= 250:
            break
    
    # Local search: try to add more sets
    if time.time() < deadline - 5:
        family = list(best_family)
        family_set = set(family)
        family_size = len(family)
        
        # Multiple rounds of trying to add random sets
        for round_num in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try up to 200 random candidates per round
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                idx = rng.randint(0, n_sets - 1)
                cand = masks[idx]
                
                if cand not in family_set:
                    if is_valid_addition(cand, family):
                        family.append(cand)
                        family_set.add(cand)
                        family_size += 1
                        improved = True
                        if family_size >= 250:
                            break
            
            if not improved:
                break
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Convert best_family back to sorted lists
    result = []
    for m in best_family:
        s = mask_to_set[m]
        result.append(sorted(s))
    
    return result

# EVOLVE-BLOCK-END
