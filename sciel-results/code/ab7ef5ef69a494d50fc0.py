import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        m = 0
        for i in combo:
            m |= (1 << i)
        all_masks.append(m)
    
    n_sets = len(all_masks)  # 1820
    
    # Precompute pairwise intersection sizes as a numpy array
    masks_arr = np.array(all_masks, dtype=np.uint32)
    # Compute all pairwise intersections
    # inter[i,j] = popcount(masks_arr[i] & masks_arr[j])
    # Use broadcasting
    inter = np.zeros((n_sets, n_sets), dtype=np.uint8)
    for i in range(n_sets):
        intersections = masks_arr & masks_arr[i]
        inter[i, :] = np.array([bin(x).count('1') for x in intersections], dtype=np.uint8)
        inter[:, i] = inter[i, :]
    
    # Actually, let's do it more efficiently with a loop but using numpy popcount
    # Python doesn't have native popcount, but we can use a lookup table or bin().count('1')
    # For 1820*1820 = 3.3M operations, this should be fine.
    
    best_family = []
    best_size = 0
    
    def greedy_with_seed(seed):
        rng = random.Random(seed)
        order = list(range(n_sets))
        rng.shuffle(order)
        chosen = []
        for idx in order:
            if len(chosen) < 2:
                chosen.append(idx)
                continue
            chosen_arr = np.array(chosen, dtype=np.int32)
            # For each pair (a,b) in chosen, check if inter[a,b] == inter[a,idx] == inter[b,idx]
            # Vectorized: get all pair intersection sizes
            # This is O(len(chosen)^2) which could be slow for large families
            # Alternative: for the new idx, compute inter[idx, chosen] array
            inter_idx = inter[idx, chosen_arr]  # size len(chosen)
            # For each pair (i,j) in chosen, check inter[chosen[i], chosen[j]] == inter_idx[i] == inter_idx[j]
            # We need to check all pairs
            n_c = len(chosen)
            # Get the submatrix of inter for chosen
            sub_inter = inter[np.ix_(chosen_arr, chosen_arr)]
            # For pair (i,j): sub_inter[i,j] == inter_idx[i] == inter_idx[j]
            # Create all pairs
            ii, jj = np.triu_indices(n_c, k=1)
            if len(ii) > 0:
                a_vals = sub_inter[ii, jj]
                b_vals = inter_idx[ii]
                c_vals = inter_idx[jj]
                if np.any((a_vals == b_vals) & (b_vals == c_vals)):
                    continue  # conflict, skip
            chosen.append(idx)
        return chosen
    
    # Try multiple randomized greedy restarts
    num_restarts = 500
    for seed in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        family = greedy_with_seed(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local improvement: try to add any remaining set
    if time.time() - start_time < DEADLINE:
        chosen_set = set(best_family)
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for idx in range(n_sets):
                if idx not in chosen_set:
                    if len(best_family) < 2:
                        best_family.append(idx)
                        chosen_set.add(idx)
                        improved = True
                        continue
                    chosen_arr = np.array(best_family, dtype=np.int32)
                    inter_idx = inter[idx, chosen_arr]
                    n_c = len(best_family)
                    sub_inter = inter[np.ix_(chosen_arr, chosen_arr)]
                    ii, jj = np.triu_indices(n_c, k=1)
                    if len(ii) > 0:
                        a_vals = sub_inter[ii, jj]
                        b_vals = inter_idx[ii]
                        c_vals = inter_idx[jj]
                        if not np.any((a_vals == b_vals) & (b_vals == c_vals)):
                            best_family.append(idx)
                            chosen_set.add(idx)
                            improved = True
                            best_size = len(best_family)
    
    # Convert to sorted lists of 1..16
    result = []
    for idx in best_family:
        m = all_masks[idx]
        elems = [i + 1 for i in range(16) if (m >> i) & 1]
        result.append(sorted(elems))
    
    return result
