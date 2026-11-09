import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    # Pre-compute all 4-element subsets of [1,16] as bitmasks
    all_rows = []
    all_masks = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i - 1))
        all_rows.append(row)
        all_masks.append(mask)
    
    n_total = len(all_masks)
    all_masks_arr = np.array(all_masks, dtype=np.uint32)
    
    # Precompute pairwise intersection matrix: inter[i,j] = all_masks[i] & all_masks[j]
    # This is a 1820x1820 matrix of uint32 - about 13 MB, very fast to build
    inter = np.zeros((n_total, n_total), dtype=np.uint32)
    for i in range(n_total):
        inter[i, i+1:] = all_masks_arr[i] & all_masks_arr[i+1:]
        inter[i+1:, i] = inter[i, i+1:]
    
    def can_add_mask(mask_idx, chosen_idx_arr):
        """Check if mask at mask_idx can be added to family given by chosen_idx_arr (array of indices)."""
        n = len(chosen_idx_arr)
        if n < 2:
            return True
        
        # Get the intersection of new mask with each chosen mask
        new_inter = all_masks_arr[mask_idx] & all_masks_arr[chosen_idx_arr]
        
        # For each pair (i,j) in chosen, check if inter[chosen[i], chosen[j]] == new_inter[i] == new_inter[j]
        # Vectorized: for each pair, compare
        ci = chosen_idx_arr
        # Get all pairwise intersections among chosen
        pair_inter = inter[ci[:, None], ci[None, :]]
        
        # Check condition: pair_inter[i,j] == new_inter[i] == new_inter[j]
        # new_inter[i] and new_inter[j] for each pair
        ni_i = new_inter[:, None]  # n x 1
        ni_j = new_inter[None, :]  # 1 x n
        
        # Condition holds where pair_inter == ni_i == ni_j
        cond = (pair_inter == ni_i) & (pair_inter == ni_j)
        # Exclude diagonal (i==j)
        np.fill_diagonal(cond, False)
        
        return not np.any(cond)
    
    def greedy_pass(order):
        chosen_idx = []
        chosen_arr = np.array([], dtype=np.int32)
        for idx in order:
            if len(chosen_idx) >= 250:
                break
            if can_add_mask(idx, chosen_arr):
                chosen_idx.append(idx)
                chosen_arr = np.append(chosen_arr, idx)
        return chosen_idx
    
    def local_search(chosen_indices, max_iters=200):
        chosen_set = set(chosen_indices)
        chosen_arr = np.array(chosen_indices, dtype=np.int32)
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > deadline - 3:
                break
            
            for out_pos in range(len(chosen_arr)):
                if time.time() > deadline - 3:
                    break
                out_idx = chosen_arr[out_pos]
                
                # Temporarily remove out_idx
                temp_arr = np.delete(chosen_arr, out_pos)
                
                for in_idx in unchosen:
                    if can_add_mask(in_idx, temp_arr):
                        # Perform swap
                        chosen_arr[out_pos] = in_idx
                        chosen_set.discard(out_idx)
                        chosen_set.add(in_idx)
                        unchosen.remove(in_idx)
                        unchosen.append(out_idx)
                        improved = True
                        break
                
                if improved:
                    break
        
        return list(chosen_arr)
    
    best_family = []
    best_size = 0
    
    indices = list(range(n_total))
    
    # Define diverse orderings
    orders = []
    
    # 1. Lexicographic
    orders.append(indices[:])
    
    # 2. Reverse lexicographic
    orders.append(indices[::-1])
    
    # 3. Sorted by bitmask value
    orders.append(sorted(indices, key=lambda i: all_masks[i]))
    
    # 4. Sorted by sum of elements
    orders.append(sorted(indices, key=lambda i: sum(all_rows[i])))
    
    # 5. Sorted by max element
    orders.append(sorted(indices, key=lambda i: all_rows[i][3]))
    
    # 6. Sorted by min element
    orders.append(sorted(indices, key=lambda i: all_rows[i][0]))
    
    # 7. Sorted by number of even elements
    orders.append(sorted(indices, key=lambda i: sum(1 for x in all_rows[i] if x % 2 == 0)))
    
    # 8-12. Random orders with different seeds
    rng = random.Random(42)
    for seed_val in [42, 123, 456, 789, 1000]:
        order_r = indices[:]
        rng2 = random.Random(seed_val)
        rng2.shuffle(order_r)
        orders.append(order_r)
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        family_indices = greedy_pass(order)
        
        # Apply local search if we have time
        if time.time() < deadline - 8:
            family_indices = local_search(family_indices, max_iters=100)
        
        family = [all_rows[i] for i in family_indices]
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        if best_size >= 250:
            break
    
    # If we still have time, do more local search on the best family
    if time.time() < deadline - 5 and best_size > 0:
        best_indices = [all_rows.index(s) for s in best_family]
        # Better: rebuild indices properly
        row_to_idx = {tuple(r): i for i, r in enumerate(all_rows)}
        best_indices = [row_to_idx[tuple(s)] for s in best_family]
        improved_indices = local_search(best_indices, max_iters=150)
        family = [all_rows[i] for i in improved_indices]
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    return [sorted(s) for s in best_family]
