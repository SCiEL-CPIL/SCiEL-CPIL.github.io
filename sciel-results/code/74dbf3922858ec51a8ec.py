"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # 1. Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_masks.append(mask)
    
    n_sets = len(all_masks)
    
    # Precompute intersection masks for all pairs?
    # intersections[i][j] = all_masks[i] & all_masks[j]
    # This is 1820x1820 matrix, ~3.3M entries. Fast to compute.
    # But we need universe_by_intersection.
    
    # 2. Precompute universe_by_intersection[S_idx][I_mask]
    # I_mask must be a subset of S.
    # We store a list of indices B such that all_masks[B] & all_masks[S] == I_mask.
    
    # To save memory/time, we only compute this on the fly or precompute for all S.
    # Precomputation:
    # For each S in range(n_sets):
    #   For each subset I of S:
    #      List B such that B & S == I
    
    # There are 1820 S's. Each has 16 subsets.
    # For each (S, I), we scan all 1820 B's.
    # Total ops: 1820 * 16 * 1820 = 53M. This is acceptable.
    
    universe_by_intersection = {}
    
    # Helper to get all subsets of a mask
    def get_subsets(mask):
        subsets = []
        sub = mask
        while sub:
            subsets.append(sub)
            sub = (sub - 1) & mask
        subsets.append(0) # Empty set
        return subsets

    # Precompute
    for s_idx in range(n_sets):
        s_mask = all_masks[s_idx]
        subs = get_subsets(s_mask)
        for i_mask in subs:
            # Find all B such that B & S == I
            valid_bs = []
            for b_idx in range(n_sets):
                if all_masks[b_idx] & s_mask == i_mask:
                    valid_bs.append(b_idx)
            universe_by_intersection[(s_idx, i_mask)] = valid_bs
            
    if time.time() > deadline:
        return []

    best_family = []
    
    # 3. Greedy Construction with Random Restarts
    # We try to build a large family.
    
    # Heuristic: Sort candidates? Random order is good for diversification.
    
    for attempt in range(20): # 20 restarts
        if time.time() > deadline:
            break
            
        candidates = list(range(n_sets))
        random.shuffle(candidates)
        
        family_indices = set()
        family_list = []
        
        for s_idx in candidates:
            if time.time() > deadline:
                break
                
            s_mask = all_masks[s_idx]
            conflict = False
            
            # Check against all A in current family
            for a_idx in family_list:
                a_mask = all_masks[a_idx]
                i_mask = s_mask & a_mask
                
                # If i_mask is not a subset of s_mask, it's impossible, but it always is.
                # Get candidates B
                key = (s_idx, i_mask)
                if key not in universe_by_intersection:
                    continue # Should not happen
                
                b_candidates = universe_by_intersection[key]
                
                # Check if any B in b_candidates is in family and B & A == i_mask
                for b_idx in b_candidates:
                    if b_idx in family_indices:
                        # Check the second condition: B & A == I
                        if all_masks[b_idx] & a_mask == i_mask:
                            conflict = True
                            break
                if conflict:
                    break
            
            if not conflict:
                family_indices.add(s_idx)
                family_list.append(s_idx)
        
        if len(family_list) > len(best_family):
            best_family = family_list[:]
            
        # Early termination if we hit a high bar?
        # Theoretical max is not known, but let's see.
        if len(best_family) > 400: # Just a guess, might be lower
            break

    # Convert indices to sorted lists of elements
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result
# EVOLVE-BLOCK-END
