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
            # Optimization: If len(members) is large, we can check pairs.
            # But usually groups are small.
            for i in range(len(members)):
                mi = members[i]
                for j in range(i+1, len(members)):
                    if (all_masks[mi] & all_masks[members[j]]) == S:
                        return False
        return True

    best_family_indices = []
    
    rng = random.Random(42)
    
    # Try multiple greedy orderings
    num_orders = 50 # Increased trials
    
    for trial in range(num_orders):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family_indices = []
        family_set = set()
        
        # Phase 1: Greedy addition in 'order'
        for idx in order:
            if time.time() > deadline:
                break
            if idx not in family_set:
                if is_valid_addition(idx, family_indices):
                    family_indices.append(idx)
                    family_set.add(idx)
        
        # Phase 2: Polish/Fill
        # Try to add any set not in family. 
        # Since adding sets only restricts future additions, one pass in a specific order is sufficient to reach a maximal set for that order.
        # We use the same 'order' for the fill phase to maintain consistency, 
        # but we could also shuffle it. Let's use the same order for simplicity and speed.
        # Actually, using a different order for fill might help escape local minima of the first phase?
        # No, the set is the same. The "maximal" property depends on the order of checking.
        # If we check in order A, we get set S_A. If we check in order B, we get S_B.
        # The greedy phase effectively did a partial fill.
        # Let's just do a full scan in the same order to ensure no set was skipped due to earlier failures?
        # Wait, if a set failed in Phase 1, it will still fail in Phase 2 because the family is larger.
        # So Phase 2 is redundant if Phase 1 iterated through all elements.
        # The only reason Phase 2 helps is if we broke out of Phase 1 early due to time?
        # Or if we want to try a *different* order for the fill?
        
        # Let's try a different strategy for Phase 2:
        # Shuffle the remaining candidates and try to add them.
        # This allows us to explore different maximal extensions.
        
        if time.time() < deadline - 2:
            remaining = [i for i in range(n_sets) if i not in family_set]
            rng.shuffle(remaining)
            for idx in remaining:
                if time.time() > deadline:
                    break
                if is_valid_addition(idx, family_indices):
                    family_indices.append(idx)
                    family_set.add(idx)
        
        if len(family_indices) > len(best_family_indices):
            best_family_indices = family_indices[:]
            
            # If we found a very good one, we might want to break early? No, keep searching.
            
    # Final polish on the best found?
    # We can try to improve best_family_indices by removing and re-adding?
    # That's more complex. Let's stick to the above.
    
    if not best_family_indices:
        # Fallback: just return empty or a small valid set
        return []
        
    return [sorted(all_sets[i]) for i in best_family_indices]
