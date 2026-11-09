"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total = len(all_masks)  # 1820
    all_mask = (1 << n) - 1
    
    # Precompute pairwise intersections for fast conflict checking
    # intersect[i][j] = all_masks[i] & all_masks[j]
    # This is 1820x1820 ~ 3.3M entries, each a small int. Memory is fine.
    intersect = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_masks[i]
        for j in range(i + 1, total):
            v = mi & all_masks[j]
            intersect[i][j] = v
            intersect[j][i] = v
    
    def can_add(family_indices, new_idx):
        """
        Check if new_idx can be added to family_indices without forming a sunflower.
        A sunflower of 3 distinct sets A, B, C has A&B == A&C == B&C.
        """
        new_mask = all_masks[new_idx]
        n_fam = len(family_indices)
        
        for i in range(n_fam):
            ai = family_indices[i]
            a_mask = all_masks[ai]
            a_inter_new = intersect[ai][new_idx]
            
            for j in range(i + 1, n_fam):
                bj = family_indices[j]
                b_mask = all_masks[bj]
                b_inter_new = intersect[bj][new_idx]
                a_inter_b = intersect[ai][bj]
                
                if a_inter_b == a_inter_new and a_inter_new == b_inter_new:
                    return False
        return True
    
    best_size = 0
    best_family = []
    
    # Strategy: Multiple random restarts with greedy selection.
    # Use different random seeds for diversity.
    rng = random.Random(12345)
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random ordering
        order = list(range(total))
        rng.shuffle(order)
        
        chosen = []
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            if can_add(chosen, idx):
                chosen.append(idx)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # Early exit if we hit the theoretical maximum
            if best_size >= 250:
                break
    
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elems = [i + 1 for i in range(n) if mask & (1 << i)]
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
