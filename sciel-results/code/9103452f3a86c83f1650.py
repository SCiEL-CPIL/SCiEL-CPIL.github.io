import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit

    # Generate all 4-element subsets of [16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)

    def is_valid_addition(fam, new_mask):
        """Check if new_mask can be added to fam without creating a sunflower."""
        n = len(fam)
        if n < 2:
            return True
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for i in range(n):
            inter = fam[i] & new_mask
            groups[inter].append(i)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if (fam[a] & fam[b]) == inter:
                        return False
        return True

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy trials with different orderings
    for trial in range(8):
        if time.time() > deadline - 35:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        for idx in order:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    family = list(best_family)
    family_set = set(family)

    # Phase 2: Try to add any remaining sets (fill-in)
    improved = True
    while improved and time.time() < deadline - 20:
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

    # Phase 3: Iterated Local Search with multiple swap attempts
    current_family = list(family)
    current_set = set(family)

    max_iterations = 20000
    for it in range(max_iterations):
        if time.time() > deadline - 3:
            break
        if not current_family:
            break

        # Pick a random set to remove
        remove_idx = rng.randrange(len(current_family))
        removed_mask = current_family[remove_idx]
        current_family.pop(remove_idx)
        current_set.discard(removed_mask)

        # Try to add a better set
        added = False
        # Try a larger sample of candidates
        n_candidates = min(1000, n_total)
        candidates = rng.sample(all_masks, n_candidates)
        for new_mask in candidates:
            if new_mask in current_set:
                continue
            if is_valid_addition(current_family, new_mask):
                current_family.append(new_mask)
                current_set.add(new_mask)
                added = True
                break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            # Restore original
            current_family.insert(remove_idx, removed_mask)
            current_set.add(removed_mask)

        if best_size >= 250:
            break

    # Phase 4: Final polish - try to add any remaining sets
    if time.time() < deadline - 2:
        family = list(best_family)
        family_set = set(family)
        for m in all_masks:
            if m in family_set:
                continue
            if is_valid_addition(family, m):
                family.append(m)
                family_set.add(m)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
