import random
import numpy as np

def construct():
    # Generate all 4-subsets of {1..16} as bitmasks
    from itertools import combinations
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    all_sets_np = np.array(all_sets, dtype=np.uint32)
    N = len(all_sets_np)
    
    best_family = []
    best_size = 0
    
    # Increase restarts
    for _ in range(300):
        indices = np.random.permutation(N)
        chosen_indices = []
        chosen_masks = np.array([], dtype=np.uint32)
        
        for s_idx in indices:
            S = all_sets_np[s_idx]
            
            if len(chosen_indices) == 0:
                chosen_indices.append(s_idx)
                chosen_masks = np.append(chosen_masks, S)
                continue
                
            # Vectorized intersection
            inters = chosen_masks & S
            
            # Check for sunflowers
            # A sunflower exists if there are i, j such that inters[i] == inters[j] == chosen_masks[i] & chosen_masks[j]
            # We group by inters
            unique_inters, counts, inverse = np.unique(inters, return_counts=True, return_inverse=True)
            
            compatible = True
            # Only check groups with count > 1
            for k in range(len(unique_inters)):
                if counts[k] > 1:
                    K = unique_inters[k]
                    # Get indices of chosen_masks that have intersection K with S
                    idxs = np.where(inverse == k)[0]
                    sub_chosen = chosen_masks[idxs]
                    
                    # Check if any pair in sub_chosen has intersection K
                    # sub_chosen is shape (m,). We need pair-wise intersections.
                    # m is small usually.
                    if len(idxs) > 1:
                        # Create pair matrix
                        # To avoid O(m^2) numpy overhead for small m, use python loop if m is very small?
                        # But numpy is generally fast.
                        pair_inters = sub_chosen[:, None] & sub_chosen[None, :]
                        # Check upper triangle
                        # We want to know if any pair_inters[i][j] == K for i < j
                        # Mask for i < j
                        m = len(idxs)
                        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
                        if np.any((pair_inters == K) & mask):
                            compatible = False
                            break
            
            if compatible:
                chosen_indices.append(s_idx)
                chosen_masks = np.append(chosen_masks, S)
        
        if len(chosen_indices) > best_size:
            best_size = len(chosen_indices)
            best_family = chosen_indices
            
    # Convert best_family (indices) to lists
    result = []
    for idx in best_family:
        mask = all_sets_np[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
