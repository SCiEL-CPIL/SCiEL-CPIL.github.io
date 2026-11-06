import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute all 4-subsets as bitmasks and sets
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Convert to bitmasks for faster intersection
    all_masks = [0] * n_sets
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_masks[i] = mask
    
    # Precompute intersections between all pairs
    # inter_mask[i][j] = bitmask of intersection of set i and set j
    # This is expensive but speeds up validation significantly
    # Actually, let's use a different approach: track for each possible intersection
    # which pairs of family members have that intersection
    
    def is_valid_addition(cand_idx, family_indices):
        """Check if cand_idx can be added to family without creating a sunflower."""
        cand_mask = all_masks[cand_idx]
        
        # Group existing family members by their intersection with cand
        groups = {}
        for idx in family_indices:
            inter = cand_mask & all_masks[idx]
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(idx)
        
        # For each intersection S, check if any two members in groups[S] 
        # also intersect to exactly S
        for S, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                mi = members[i]
                for j in range(i+1, len(members)):
                    if (all_masks[mi] & all_masks[members[j]]) == S:
                        return False
        return True
    
    best_family = []
    best_family_indices = []
    
    rng = random.Random(42)
    
    # Try multiple greedy orderings
    num_orders = 20
    for trial in range(num_orders):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family_indices = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if is_valid_addition(idx, family_indices):
                family_indices.append(idx)
        
        if len(family_indices) > len(best_family_indices):
            best_family_indices = family_indices[:]
    
    # Local search: try to improve
    if time.time() < deadline - 5:
        family_indices = list(best_family_indices)
        family_set = set(family_indices)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try to add a random set
            improved = False
            for _ in range(200):
                idx = rng.randint(0, n_sets - 1)
                if idx not in family_set:
                    if is_valid_addition(idx, family_indices):
                        family_indices.append(idx)
                        family_set.add(idx)
                        improved = True
                        break
            
            if not improved:
                break
    
    return [sorted(all_sets[i]) for i in best_family_indices if i in set(family_indices)] if len(family_indices) > len(best_family_indices) else [sorted(all_sets[i]) for i in best_family_indices]
