import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Returns a set of integer points in 11D where min pairwise dist sq >= max norm sq.
    Uses a hybrid Greedy + Local Search approach on a larger candidate pool.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit
    
    d = 11
    
    # Generate candidates with squared norm <= 8
    candidates = []
    
    # Helper to add a vector
    def add_vec(v):
        candidates.append(v)
        
    # Norm 1: (±1, 0, ...)
    for i in range(d):
        for sign in [1, -1]:
            v = [0] * d
            v[i] = sign
            add_vec(v)
            
    # Norm 2: (±1, ±1, 0, ...)
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    add_vec(v)
                    
    # Norm 4: (±2, 0, ...) or (±1, ±1, ±1, ±1, 0, ...)
    for i in range(d):
        for sign in [1, -1]:
            v = [0] * d
            v[i] = 2 * sign
            add_vec(v)
            
    for idx in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = [0] * d
                        v[idx[0]] = s1
                        v[idx[1]] = s2
                        v[idx[2]] = s3
                        v[idx[3]] = s4
                        add_vec(v)
                        
    # Norm 5: (±1, ±2, 0, ...)
    for i in range(d):
        for j in range(d):
            if i == j: continue
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = 2 * s2
                    add_vec(v)

    # Norm 8: (±2, ±2, 0, ...) or (±2, ±1, ±1, 0, ...) -- wait, 4+1+1=6. 
    # Let's stick to norms 1, 2, 4, 5 for now to keep candidate count reasonable (~6000).
    # If we add norm 8: (±2, ±2, 0...) -> 11*10*2*2 = 440.
    # (±2, ±1, ±1, 0...) -> 11*10*9 * 2*2*2? No. 
    # Let's just use norms 1, 2, 4, 5.
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute pairwise squared distances
    # dist_sq[i, j] = ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i . c_j
    # We can compute dot products first
    dot_matrix = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_matrix
    
    # Ensure diagonal is 0 (distance to self)
    np.fill_diagonal(dist_sq, 0)
    
    best_selected = []
    best_max_norm = 0
    
    # Strategy 1: Greedy initialization
    # Sort candidates by norm (lower norm is "better" for packing)
    order = np.argsort(norms_sq)
    
    selected = []
    max_norm = 0
    
    # Greedy: pick lowest norm, then next lowest that fits
    # This is O(N^2) if done naively, but N~6000, so 36M ops. Acceptable.
    remaining = np.ones(n_cand, dtype=bool)
    
    # To speed up, we can just iterate through sorted candidates
    # and check against current selected set.
    
    current_selected = []
    current_max_norm = 0
    
    for idx in order:
        if time.time() > deadline - 100:
            break
            
        # Check if candidate idx can be added
        # It must satisfy: dist(idx, j) >= max(current_max_norm, norms_sq[idx]) for all j in current_selected
        # Actually, the condition is: min_dist >= max_norm.
        # If we add idx, new_max_norm = max(current_max_norm, norms_sq[idx])
        # We need dist(idx, j) >= new_max_norm for all j in current_selected.
        
        new_max_norm = max(current_max_norm, norms_sq[idx])
        
        if not current_selected:
            current_selected.append(idx)
            current_max_norm = new_max_norm
            continue
            
        # Check distances to all currently selected
        sel_indices = np.array(current_selected)
        dists = dist_sq[idx, sel_indices]
        
        if np.min(dists) >= new_max_norm:
            current_selected.append(idx)
            current_max_norm = new_max_norm
            
    if len(current_selected) > len(best_selected):
        best_selected = current_selected[:]
        best_max_norm = current_max_norm

    # Strategy 2: Local Search (Hill Climbing)
    # Start from best_selected
    # Try to add any non-selected candidate.
    # If it fails, try to swap with a conflicting candidate.
    
    selected_set = set(best_selected)
    current_max_norm = best_max_norm
    
    # Precompute which candidates are selected
    is_selected = np.zeros(n_cand, dtype=bool)
    for idx in best_selected:
        is_selected[idx] = True
        
    # List of non-selected candidates
    non_selected = np.where(~is_selected)[0]
    
    # Shuffle non-selected to break symmetry
    rng = np.random.RandomState(42)
    rng.shuffle(non_selected)
    
    improved = True
    while improved and time.time() < deadline - 50:
        improved = False
        
        for idx in non_selected:
            if time.time() > deadline - 50:
                break
                
            new_max_norm = max(current_max_norm, norms_sq[idx])
            
            # Check if idx can be added
            # We need dist(idx, j) >= new_max_norm for all j in selected_set
            # Since selected_set is a list, we can use dist_sq[idx, list]
            sel_list = list(selected_set)
            if not sel_list:
                selected_set.add(idx)
                is_selected[idx] = True
                non_selected = np.delete(non_selected, np.where(non_selected == idx)[0])
                current_max_norm = new_max_norm
                improved = True
                continue
                
            dists = dist_sq[idx, sel_list]
            min_dist = np.min(dists)
            
            if min_dist >= new_max_norm:
                # Can add
                selected_set.add(idx)
                is_selected[idx] = True
                non_selected = np.delete(non_selected, np.where(non_selected == idx)[0])
                current_max_norm = new_max_norm
                improved = True
            else:
                # Cannot add directly. Try to swap.
                # Find the point j in selected_set that causes the conflict (min_dist < new_max_norm)
                # Actually, any point with dist < new_max_norm is a conflict.
                # We want to remove a point j and add idx.
                # New set = (selected_set - {j}) U {idx}
                # New max norm = max(norms_sq of new set)
                # This is complex to check efficiently for all j.
                
                # Heuristic: Find the point j in selected_set that is "closest" to idx
                # or has the largest norm?
                # Let's try swapping with the point j that has the minimum distance to idx.
                # If dist(idx, j) is small, removing j might help if j was "blocking" idx.
                # But we also need to check if removing j and adding idx keeps the rest valid.
                
                # Let's identify all conflicting points
                conflict_mask = dists < new_max_norm
                if not np.any(conflict_mask):
                    continue
                    
                conflicting_indices = [sel_list[k] for k in range(len(sel_list)) if conflict_mask[k]]
                
                # Try swapping with each conflicting point (limit to first few to save time)
                for j in conflicting_indices[:3]: # Try up to 3 swaps
                    # Proposed new set
                    # Remove j, add idx
                    # Check if new set is valid
                    # New max norm: max(current_max_norm excluding j, norms_sq[idx])
                    # If j was the unique max norm point, max_norm might decrease.
                    
                    # Let's compute new_max_norm carefully
                    if norms_sq[j] == current_max_norm:
                        # Need to find next max norm in remaining set
                        # This is expensive. Approximate: assume current_max_norm stays or decreases.
                        # For safety, let's just check if the new set satisfies the condition.
                        pass
                    
                    # Efficient check:
                    # The condition is min_pairwise_dist >= max_norm.
                    # When we remove j and add idx:
                    # 1. Distances between remaining points are unchanged.
                    # 2. Distances from idx to remaining points must be >= new_max_norm.
                    # 3. new_max_norm = max(norms_sq of all points in new set).
                    
                    # Let's estimate new_max_norm
                    # It is at most max(current_max_norm, norms_sq[idx])
                    # It is at least max(norms_sq[idx], max(norms_sq of remaining points))
                    
                    # To be safe, let's just construct the new list and check?
                    # That is O(k). k is ~200. 
                    # Doing this for 3 swaps per candidate (6000 candidates) is 6000 * 3 * 200 = 3.6M ops.
                    # This is fine.
                    
                    new_selected_list = [x for x in sel_list if x != j]
                    new_selected_list.append(idx)
                    
                    # Compute new max norm
                    new_max_norm_check = max([norms_sq[x] for x in new_selected_list])
                    
                    # Check min distance
                    # We need min dist among new_selected_list >= new_max_norm_check
                    # We can reuse dist_sq
                    # min_dist_new = min( min_dist_among_remaining, min_dist_from_idx_to_remaining )
                    # min_dist_among_remaining was valid before (was >= old_max_norm).
                    # If new_max_norm_check <= old_max_norm, then min_dist_among_remaining is still valid.
                    # If new_max_norm_check > old_max_norm, we need to re-check?
                    # No, the old set was valid with old_max_norm.
                    # The distances among remaining points are the same.
                    # If new_max_norm_check > old_max_norm, it's possible that some pair in remaining
                    # had distance < new_max_norm_check.
                    # However, since old set was valid, min_dist_remaining >= old_max_norm.
                    # If new_max_norm_check is larger, we might have a violation.
                    
                    # Let's just check all pairs in the new set? O(k^2) is too slow.
                    # Let's check:
                    # 1. min(dist(idx, r) for r in remaining) >= new_max_norm_check
                    # 2. min(dist(r1, r2) for r1, r2 in remaining) >= new_max_norm_check
                    
                    # We know min(dist(r1, r2)) >= old_max_norm.
                    # If new_max_norm_check <= old_max_norm, condition 2 holds.
                    # If new_max_norm_check > old_max_norm, we need to verify.
                    # But new_max_norm_check is max(norms_sq[idx], max_norms_remaining).
                    # If norms_sq[idx] > old_max_norm, then new_max_norm_check = norms_sq[idx].
                    # We need min_dist_remaining >= norms_sq[idx].
                    
                    # Let's compute min_dist_remaining.
                    # We can cache the min_dist of the current set?
                    # It's dynamic.
                    
                    # Simplification: Only attempt swap if norms_sq[idx] <= current_max_norm.
                    # This ensures new_max_norm_check <= current_max_norm.
                    # Then condition 2 holds automatically.
                    # We only need to check condition 1.
                    
                    if norms_sq[idx] <= current_max_norm:
                        # new_max_norm_check is either current_max_norm (if j was not the unique max) 
                        # or the second largest norm.
                        # In any case, new_max_norm_check <= current_max_norm.
                        # So we just need to check dist(idx, r) >= new_max_norm_check for all r in remaining.
                        
                        # To be safe, let's use current_max_norm as the threshold.
                        # If dist(idx, r) >= current_max_norm for all r in remaining, 
                        # then dist(idx, r) >= new_max_norm_check is guaranteed.
                        
                        remaining_indices = np.array([x for x in sel_list if x != j])
                        if len(remaining_indices) > 0:
                            dists_to_remaining = dist_sq[idx, remaining_indices]
                            if np.min(dists_to_remaining) >= current_max_norm:
                                # Swap is valid
                                selected_set.remove(j)
                                is_selected[j] = False
                                selected_set.add(idx)
                                is_selected[idx] = True
                                non_selected = np.delete(non_selected, np.where(non_selected == idx)[0])
                                non_selected = np.append(non_selected, j)
                                # Update current_max_norm
                                # It might have decreased if j was the max.
                                # Recalculate
                                current_max_norm = max([norms_sq[x] for x in selected_set])
                                improved = True
                                break # Break swap loop, try next candidate
    
    # Final result
    if not best_selected:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = candidates[best_selected]
    return np.around(result).astype(np.int64)
