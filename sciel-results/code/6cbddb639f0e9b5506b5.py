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
    
    # Precompute pairwise intersections: inter_table[i][j] = all_masks[i] & all_masks[j]
    # Use a 2D list for fast lookup
    inter_table = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter_table[i][j] = ij
            inter_table[j][i] = ij
    
    # Map mask to index for quick lookup
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}

    def is_valid_addition(fam_indices, new_idx):
        """Check if all_masks[new_idx] can be added to fam (given as indices) without creating a sunflower."""
        n = len(fam_indices)
        if n < 2:
            return True
        
        new_mask = all_masks[new_idx]
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for i in range(n):
            fi = fam_indices[i]
            inter = inter_table[fi][new_idx]
            groups[inter].append(fi)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if inter_table[a][b] == inter:
                        return False
        return True

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy trials with different orderings
    for trial in range(12):
        if time.time() > deadline - 30:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        for idx in order:
            if is_valid_addition(family, idx):
                family.append(idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    family = list(best_family)
    family_set = set(family)

    # Phase 2: Try to add any remaining sets (fill-in)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for idx in range(n_total):
            if idx in family_set:
                continue
            if is_valid_addition(family, idx):
                family.append(idx)
                family_set.add(idx)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Phase 3: Iterated Local Search with swap attempts
    current_family = list(family)
    current_set = set(family)

    max_iterations = 30000
    for it in range(max_iterations):
        if time.time() > deadline - 3:
            break
        if not current_family:
            break

        # Pick a random set to remove
        remove_idx = rng.randrange(len(current_family))
        removed_idx = current_family[remove_idx]
        current_family.pop(remove_idx)
        current_set.discard(removed_idx)

        # Try to add a better set - sample candidates
        added = False
        n_candidates = min(2000, n_total)
        candidates = rng.sample(range(n_total), n_candidates)
        for new_idx in candidates:
            if new_idx in current_set:
                continue
            if is_valid_addition(current_family, new_idx):
                current_family.append(new_idx)
                current_set.add(new_idx)
                added = True
                break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            # Restore original
            current_family.insert(remove_idx, removed_idx)
            current_set.add(removed_idx)

        if best_size >= 250:
            break

    # Phase 4: Final polish - try to add any remaining sets
    if time.time() < deadline - 2:
        family = list(best_family)
        family_set = set(family)
        for idx in range(n_total):
            if idx in family_set:
                continue
            if is_valid_addition(family, idx):
                family.append(idx)
                family_set.add(idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break

    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
