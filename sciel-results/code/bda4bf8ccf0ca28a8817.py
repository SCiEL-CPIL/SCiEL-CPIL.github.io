import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute all 4-subsets of [1,16]
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute frozenset representations
    all_frozensets = [frozenset(s) for s in all_sets]
    
    # Precompute all pairwise intersections as bitmasks for speed
    # Each 4-subset of [1,16] -> bitmask of 16 bits
    all_bitmasks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_bitmasks.append(mask)
    
    # Precompute intersection bitmask for all pairs
    # inter_masks[i] is a list of n_sets ints: inter_masks[i][j] = bitmask of all_sets[i] & all_sets[j]
    # But that's 1820*1820 = 3.3M entries. Let's just compute on the fly using & operation.
    
    # Build conflict graph: two subsets conflict if there exists a third subset that
    # together with them forms a sunflower (all three pairwise intersections equal).
    # Actually, the constraint is: no THREE distinct sets have all three pairwise intersections equal.
    # So the "conflict" is a triple, not a pair. We need a different approach.
    
    # For maximum independent set on a 3-uniform hypergraph, we can use:
    # - Greedy construction with smart ordering
    # - Local search (add/remove swaps)
    # - Branch and bound on promising candidates
    
    def is_valid_addition(cand_mask, family_masks):
        """Check if cand_mask can be added to family_masks without creating a sunflower."""
        # Group family members by their intersection with cand
        groups = {}
        for fm in family_masks:
            inter = cand_mask & fm
            if inter not in groups:
                groups[inter] = [fm]
            else:
                groups[inter].append(fm)
        
        # For each group with intersection S, check if any pair has intersection S
        for S, members in groups.items():
            if len(members) < 2:
                continue
            n = len(members)
            for a in range(n):
                for b in range(a + 1, n):
                    if (members[a] & members[b]) == S:
                        return False
        return True
    
    def greedy_construct(order, family_masks, family_indices):
        """Add sets from order to family, skipping invalid ones."""
        for idx in order:
            if not is_valid_addition(all_bitmasks[idx], family_masks):
                continue
            family_masks.append(all_bitmasks[idx])
            family_indices.append(idx)
    
    def local_search_add(family_masks, family_indices, rng, max_iters=500):
        """Try to add more sets via random sampling."""
        added = 0
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in family_indices:
                continue
            if is_valid_addition(all_bitmasks[idx], family_masks):
                family_masks.append(all_bitmasks[idx])
                family_indices.append(idx)
                added += 1
        return added
    
    def local_search_swap(family_masks, family_indices, rng, max_iters=200):
        """Try swap operations: remove one set, add up to two others."""
        improved = 0
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Pick a random set to remove
            rem_idx = rng.randint(0, len(family_masks) - 1)
            removed_mask = family_masks[rem_idx]
            removed_idx = family_indices[rem_idx]
            
            # Temporarily remove
            family_masks[rem_idx] = None
            family_indices[rem_idx] = -1
            
            # Try to add up to 2 new sets
            added_new = []
            for _ in range(3):
                cand_idx = rng.randint(0, n_sets - 1)
                if cand_idx == removed_idx:
                    continue
                if any(cand_idx == fi for fi in family_indices if fi >= 0):
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], [m for m in family_masks if m is not None]):
                    family_masks.append(all_bitmasks[cand_idx])
                    family_indices.append(cand_idx)
                    added_new.append(cand_idx)
            
            # Check if we improved
            new_size = len([m for m in family_masks if m is not None])
            old_size = len(family_masks) - 1  # before we added new ones, minus the removed
            if new_size > len(family_masks) - 1 - len(added_new) + len(added_new):
                # We gained sets
                improved += 1
            else:
                # Revert: remove the newly added, restore the removed
                for _ in added_new:
                    family_masks.pop()
                    family_indices.pop()
                family_masks[rem_idx] = removed_mask
                family_indices[rem_idx] = removed_idx
        
        return improved
    
    rng = random.Random(42)
    best_family_indices = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        elif trial == 1:
            order = list(range(n_sets))
            order.reverse()
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family_masks = []
        family_indices = []
        greedy_construct(order, family_masks, family_indices)
        
        # Local search to add more
        local_search_add(family_masks, family_indices, rng, max_iters=300)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family_indices = list(family_indices)
    
    # Phase 2: Local search swaps on the best family
    if time.time() < deadline - 5 and best_family_indices:
        family_masks = [all_bitmasks[i] for i in best_family_indices]
        family_indices = list(best_family_indices)
        
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            local_search_swap(family_masks, family_indices, rng, max_iters=100)
            local_search_add(family_masks, family_indices, rng, max_iters=200)
            
            if len(family_indices) > best_size:
                best_size = len(family_indices)
                best_family_indices = list(family_indices)
    
    # Phase 3: More aggressive local search if time permits
    if time.time() < deadline - 3 and best_family_indices:
        family_masks = [all_bitmasks[i] for i in best_family_indices]
        family_indices = list(best_family_indices)
        
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in family_indices:
                continue
            if is_valid_addition(all_bitmasks[idx], family_masks):
                family_masks.append(all_bitmasks[idx])
                family_indices.append(idx)
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family_indices = list(family_indices)
    
    return [sorted(all_sets[i]) for i in best_family_indices]
