import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55

    # Generate all 4-subsets of [16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)
    rng = random.Random(42)

    def is_valid_addition(fam, new_mask):
        n = len(fam)
        if n < 2:
            return True
        
        # Calculate intersections of new_mask with all family members
        inters = [0] * n
        for i in range(n):
            inters[i] = fam[i] & new_mask
        
        # Group indices by their intersection value
        groups = defaultdict(list)
        for i, v in enumerate(inters):
            groups[v].append(i)
            
        # Check for sunflower condition
        for v, indices in groups.items():
            if len(indices) < 2:
                continue
            # If there are multiple sets in fam that intersect new_mask at v,
            # check if any pair among them also intersects each other at v.
            for idx_a in range(len(indices)):
                i = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    j = indices[idx_b]
                    if (fam[i] & fam[j]) == v:
                        return False
        return True

    best_family = []
    best_size = 0

    # Phase 1: Multiple Greedy Trials
    for trial in range(10):
        if time.time() > deadline - 30:
            break
            
        order = list(range(n_total))
        rng.shuffle(order)
        
        family = []
        for idx in order:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    # Phase 2: Local Search (Swap)
    current_family = list(best_family)
    current_set = set(current_family)
    
    # Pre-compute a list of all masks for random sampling
    all_masks_list = all_masks
    
    for step in range(5000):
        if time.time() > deadline - 5:
            break
        if not current_family:
            break
            
        # Remove a random element
        remove_idx = rng.randrange(len(current_family))
        removed_mask = current_family[remove_idx]
        current_family.pop(remove_idx)
        current_set.discard(removed_mask)
        
        added = False
        # Try to add a random element
        # We can sample from all masks not in current_set
        # To avoid O(N) check for "in current_set", we can just sample and check
        # But checking is_valid_addition is O(N^2).
        
        # Try a few candidates
        candidates_found = 0
        while candidates_found < 50:
            new_mask = rng.choice(all_masks_list)
            if new_mask in current_set:
                continue
                
            if is_valid_addition(current_family, new_mask):
                current_family.append(new_mask)
                current_set.add(new_mask)
                added = True
                break
            candidates_found += 1
            
        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            # Revert
            current_family.insert(remove_idx, removed_mask)
            current_set.add(removed_mask)
            
        if best_size >= 250:
            break

    # Phase 3: Final Polish (Try to add any remaining set)
    if time.time() < deadline - 2:
        family = list(best_family)
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for m in all_masks:
                if m in family_set:
                    continue
                if is_valid_addition(family, m):
                    family.append(m)
                    family_set.add(m)
                    improved = True
                    if len(family) > best_size:
                        best_size = len(family)
                        best_family = list(family)
                    if len(family) >= 250:
                        break
            if len(family) >= 250:
                break

    # Convert back to list of lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
