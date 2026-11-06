# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 16
    k = 4
    all_sets = [set(row) for row in combinations(range(1, n+1), k)]
    total = len(all_sets)  # 1820
    
    # Convert to bitmask representation for speed
    # Each set is a 16-bit integer
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x-1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    
    # Greedy approach with randomized restarts
    import random
    random.seed(42)
    
    best_family = []
    
    def check_sunflower(family_masks, new_mask):
        """Check if adding new_mask creates a 3-sunflower with any two in family."""
        # For each pair (i,j) in family, check if family[i]&family[j] == family[i]&new_mask == family[j]&new_mask
        # Optimize: group by intersection with new_mask
        # If A∩S = I and B∩S = I, then we need A∩B = I
        # So: for each possible intersection value I, collect all A in family where A∩S = I
        # Then check if any two in that group have A∩B = I
        
        n_fam = len(family_masks)
        if n_fam < 2:
            return False
        
        # Group family members by their intersection with new_mask
        groups = {}
        for i in range(n_fam):
            inter = family_masks[i] & new_mask
            if inter not in groups:
                groups[inter] = [i]
            else:
                groups[inter].append(i)
        
        # For each group, check if any two members have mutual intersection equal to the group key
        for inter, members in groups.items():
            if len(members) < 2:
                continue
            for idx_a in range(len(members)):
                a = members[idx_a]
                for idx_b in range(idx_a+1, len(members)):
                    b = members[idx_b]
                    if family_masks[a] & family_masks[b] == inter:
                        return True
        return False
    
    # Try multiple random orderings
    best_size = 0
    best_result = []
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        order = list(range(total))
        random.shuffle(order)
        
        family = []
        family_masks = []
        
        for idx in order:
            if time.time() > deadline:
                break
            m = masks[idx]
            if not check_sunflower(family_masks, m):
                family.append(all_sets[idx])
                family_masks.append(m)
        
        if len(family) > best_size:
            best_size = len(family)
            best_result = [sorted(s) for s in family]
    
    return best_result
# EVOLVE-BLOCK-END
