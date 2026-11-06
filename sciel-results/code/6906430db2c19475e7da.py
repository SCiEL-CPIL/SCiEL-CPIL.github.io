import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate all 4-subsets of [1, 16]
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n = len(all_sets)
    
    # Convert to bitmasks (0-indexed bits 0..15)
    set_masks = []
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        set_masks.append(m)
        
    # 2. Precompute pairwise core and forbidden masks
    # core[i][j] = set_masks[i] & set_masks[j]
    # forbid[i][j] = (set_masks[i] ^ core[i][j]) | (set_masks[j] ^ core[i][j])
    # Note: A \ B = A ^ (A & B). So (A \ K) | (B \ K) = (A ^ K) | (B ^ K)
    # Since K = A & B, A ^ K = A \ K.
    
    core_masks = [[0] * n for _ in range(n)]
    forbid_masks = [[0] * n for _ in range(n)]
    
    for i in range(n):
        mi = set_masks[i]
        for j in range(i + 1, n):
            mj = set_masks[j]
            k = mi & mj
            f = (mi ^ k) | (mj ^ k)
            core_masks[i][j] = k
            core_masks[j][i] = k
            forbid_masks[i][j] = f
            forbid_masks[j][i] = f
            
    # 3. Helper function to check if a mask C can be added to a family of indices
    def is_valid(c_mask, family_indices):
        # Check against all pairs in family
        # If family size < 2, it's valid
        if len(family_indices) < 2:
            return True
            
        # Optimization: iterate through pairs
        # To speed up, we can just check all pairs.
        # family_indices is a list.
        for a_idx in range(len(family_indices)):
            i = family_indices[a_idx]
            for b_idx in range(a_idx + 1, len(family_indices)):
                j = family_indices[b_idx]
                # Condition for sunflower:
                # c_mask must contain core[i][j]
                # c_mask must NOT contain any part of forbid[i][j]
                if (c_mask & core_masks[i][j]) == core_masks[i][j]:
                    if (c_mask & forbid_masks[i][j]) == 0:
                        return False
        return True

    best_family = []
    best_size = 0
    
    # 4. Main Search Loop: Multiple Randomized Greedy Restarts
    while time.time() < deadline:
        family = []
        family_set = set()
        
        # Random order
        indices = list(range(n))
        random.shuffle(indices)
        
        # Greedy Pass
        for idx in indices:
            if idx in family_set:
                continue
            c_mask = set_masks[idx]
            if is_valid(c_mask, family):
                family.append(idx)
                family_set.add(idx)
                
        # Local Improvement: Try to add any remaining set
        # This is redundant if greedy was perfect, but greedy might have skipped
        # a set that became valid? No, adding sets only adds constraints.
        # So if it was invalid, it remains invalid.
        # However, we might have missed a set that was valid but skipped due to order?
        # No, we check ALL sets in the shuffled list.
        # So the greedy result is a Maximal Independent Set (MIS).
        
        # To find larger MIS, we can try 1-swap or 2-swap.
        # Let's try a simple 1-swap: Remove one, add one? No, size stays same.
        # We need to Remove one, Add TWO?
        
        # Let's try a "kick" phase:
        # Pick a random element in family, remove it, and try to fill the space
        # with as many sets as possible from the pool.
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try to add any set not in family
            # (This is actually useless as established, but cheap to check if we implement it differently)
            # Actually, let's just do random restarts. They are very effective for MIS.
            
            # Let's do a limited 1-swap local search to escape local optima?
            # Pick a random i in family.
            # Remove i.
            # Try to add any j not in family.
            # If we can add j, family size is same.
            # But maybe removing i allows adding j AND k?
            # Let's try:
            if family:
                # Pick random victim
                victim_pos = random.randint(0, len(family) - 1)
                victim_idx = family[victim_pos]
                
                # Create temp family without victim
                temp_family = family[:victim_pos] + family[victim_pos+1:]
                temp_set = family_set - {victim_idx}
                
                # Try to add sets from the pool
                added_count = 0
                new_family = temp_family.copy()
                new_set = temp_set.copy()
                
                # We want to add as many as possible?
                # Just try to add ONE set that was blocked by victim?
                # Or just try to add any valid set.
                
                # To be efficient, we can just try to add one set.
                # If we can add one, we haven't improved size.
                # We need to add MORE than we removed.
                # So we must add at least 2 if we remove 1.
                
                # Let's try to add 2 sets.
                # This is expensive.
                
                # Alternative: Just rely on many restarts.
                # Let's break and do another restart if we don't improve quickly.
                pass
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
    # Convert best_family to list of sets
    result = []
    for idx in best_family:
        result.append(sorted(all_sets[idx]))
        
    return result
