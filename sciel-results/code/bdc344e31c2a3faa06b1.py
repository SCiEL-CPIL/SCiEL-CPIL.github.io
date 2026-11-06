import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)
    
    # Precompute pairwise intersections as a matrix for O(1) lookup
    # inter_matrix[i][j] = all_masks[i] & all_masks[j]
    inter_matrix = np.zeros((n_total, n_total), dtype=np.int32)
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            v = mi & all_masks[j]
            inter_matrix[i, j] = v
            inter_matrix[j, i] = v
    
    def is_valid_addition(fam_indices, new_idx):
        """Check if adding new_idx to fam_indices (list of indices into all_masks) is valid."""
        n = len(fam_indices)
        if n < 2:
            return True
        
        # For each pair (a, b) in fam, check if the sunflower condition is violated with new
        # Condition: there exist distinct a, b in fam such that:
        #   (fam[a] & new) == (fam[b] & new) == (fam[a] & fam[b])
        # This is equivalent to:
        #   inter_matrix[a][new] == inter_matrix[b][new] == inter_matrix[a][b]
        
        # Group existing family members by their intersection with new
        groups = defaultdict(list)
        for idx in fam_indices:
            v = inter_matrix[idx, new_idx]
            groups[v].append(idx)
        
        for v, indices in groups.items():
            if len(indices) < 2:
                continue
            # Check all pairs in this group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a = indices[i]
                    b = indices[j]
                    if inter_matrix[a, b] == v:
                        return False
        return True

    best_family = []
    best_size = 0
    
    # Multiple greedy trials with different orderings
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        
        family_indices = []
        for idx in order:
            if is_valid_addition(family_indices, idx):
                family_indices.append(idx)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family = list(family_indices)
    
    # Convert to mask list for further processing
    family = [all_masks[i] for i in best_family]
    family_set = set(family)
    
    # Try to add any remaining sets
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for idx in range(n_total):
            m = all_masks[idx]
            if m in family_set:
                continue
            if is_valid_addition(family_indices, idx):
                family_indices.append(idx)
                family.append(m)
                family_set.add(m)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family_indices)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break
    
    # Iterated Local Search: Swap moves
    current_family_indices = list(family_indices)
    current_set = set(all_masks[i] for i in current_family_indices)
    
    for it in range(10000):
        if time.time() > deadline - 2:
            break
        if not current_family_indices:
            break
        
        # Pick a random set to remove
        remove_pos = rng.randrange(len(current_family_indices))
        removed_idx = current_family_indices[remove_pos]
        current_family_indices.pop(remove_pos)
        current_set.discard(all_masks[removed_idx])
        
        # Try to add a random set from the pool
        added = False
        n_candidates = min(1000, n_total)
        candidates = rng.sample(range(n_total), n_candidates)
        for new_idx in candidates:
            if all_masks[new_idx] in current_set:
                continue
            if is_valid_addition(current_family_indices, new_idx):
                current_family_indices.append(new_idx)
                current_set.add(all_masks[new_idx])
                added = True
                break
        
        if added:
            if len(current_family_indices) > best_size:
                best_size = len(current_family_indices)
                best_family = list(current_family_indices)
        else:
            # Restore original
            current_family_indices.insert(remove_pos, removed_idx)
            current_set.add(all_masks[removed_idx])
        
        if best_size >= 250:
            break
    
    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result

if __name__ == "__main__":
    # Just for testing, not part of the evaluation call
    res = construct()
    print(len(res))
