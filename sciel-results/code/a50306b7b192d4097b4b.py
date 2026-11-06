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
    
    # Precompute pairwise intersections for all mask pairs
    inter_matrix = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter_matrix[i][j] = ij
            inter_matrix[j][i] = ij
    
    def is_valid_addition_fast(fam_indices, new_idx):
        """Check if all_masks[new_idx] can be added to family (given as indices) without creating a sunflower."""
        n = len(fam_indices)
        if n < 2:
            return True
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for fam_idx in fam_indices:
            inter = inter_matrix[fam_idx][new_idx]
            groups[inter].append(fam_idx)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            # Early exit optimization: if group is large, check strategically
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if inter_matrix[a][b] == inter:
                        return False
        return True

    best_family = []
    best_size = 0
    best_indices = []

    # Phase 1: Multiple greedy trials with different orderings
    for trial in range(30):
        if time.time() > deadline - 45:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family_indices = []
        family_masks = []
        for idx in order:
            if is_valid_addition_fast(family_indices, idx):
                family_indices.append(idx)
                family_masks.append(all_masks[idx])
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family = list(family_masks)
            best_indices = list(family_indices)

    family = list(best_family)
    family_set = set(family)
    family_indices = list(best_indices)

    # Phase 2: Try to add any remaining sets (fill-in)
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for idx in range(n_total):
            if all_masks[idx] in family_set:
                continue
            if is_valid_addition_fast(family_indices, idx):
                family.append(all_masks[idx])
                family_set.add(all_masks[idx])
                family_indices.append(idx)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                    best_indices = list(family_indices)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Phase 3: Iterated Local Search with rejected-set tracking
    current_family = list(family)
    current_set = set(family)
    current_indices = list(family_indices)
    current_idx_set = set(current_indices)
    
    # Track rejected sets for faster local search
    rejected = set()
    for idx in range(n_total):
        if all_masks[idx] not in current_set:
            rejected.add(idx)

    no_improve_count = 0
    max_no_improve = 3000
    while time.time() < deadline - 3:
        if not current_family:
            break

        # Pick a random set to remove
        remove_pos = rng.randrange(len(current_family))
        removed_idx = current_indices[remove_pos]
        removed_mask = current_family[remove_pos]
        current_family.pop(remove_pos)
        current_set.discard(removed_mask)
        current_indices.pop(remove_pos)
        current_idx_set.discard(removed_idx)

        # Try to add from rejected sets first (they were close before)
        added = False
        # Shuffle rejected for variety
        rejected_list = list(rejected)
        rng.shuffle(rejected_list)
        for new_idx in rejected_list:
            if is_valid_addition_fast(current_indices, new_idx):
                current_family.append(all_masks[new_idx])
                current_set.add(all_masks[new_idx])
                current_indices.append(new_idx)
                current_idx_set.add(new_idx)
                rejected.discard(new_idx)
                added = True
                break

        # If no rejected set works, try random non-family sets
        if not added:
            non_family = [i for i in range(n_total) if all_masks[i] not in current_set]
            if non_family:
                candidates = rng.sample(non_family, min(200, len(non_family)))
                for new_idx in candidates:
                    if is_valid_addition_fast(current_indices, new_idx):
                        current_family.append(all_masks[new_idx])
                        current_set.add(all_masks[new_idx])
                        current_indices.append(new_idx)
                        current_idx_set.add(new_idx)
                        rejected.discard(new_idx)
                        added = True
                        break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
                best_indices = list(current_indices)
            no_improve_count = 0
            # Update rejected set: remove newly added, add removed
            rejected.discard(removed_idx)  # already removed from family
            # The removed index is now available again
            rejected.add(removed_idx)
        else:
            # Restore original
            current_family.insert(remove_pos, removed_mask)
            current_set.add(removed_mask)
            current_indices.insert(remove_pos, removed_idx)
            current_idx_set.add(removed_idx)
            no_improve_count += 1
            # If stuck for too long, restart with a fresh greedy
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                order = list(range(n_total))
                rng.shuffle(order)
                current_family = []
                current_set = set()
                current_indices = []
                current_idx_set = set()
                for idx in order:
                    if is_valid_addition_fast(current_indices, idx):
                        current_family.append(all_masks[idx])
                        current_set.add(all_masks[idx])
                        current_indices.append(idx)
                        current_idx_set.add(idx)
                rejected = set()
                for idx in range(n_total):
                    if all_masks[idx] not in current_set:
                        rejected.add(idx)
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = list(current_family)
                    best_indices = list(current_indices)

        if best_size >= 250:
            break

    # Phase 4: Final polish - try to add any remaining sets
    if time.time() < deadline - 2:
        family = list(best_family)
        family_set = set(family)
        family_indices = list(best_indices)
        for idx in range(n_total):
            if all_masks[idx] in family_set:
                continue
            if is_valid_addition_fast(family_indices, idx):
                family.append(all_masks[idx])
                family_set.add(all_masks[idx])
                family_indices.append(idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                    best_indices = list(family_indices)
                if len(family) >= 250:
                    break

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
