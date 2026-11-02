import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit (or 150s)
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280... 
    # Wait, C(11,4) = 330, 2^4 = 16, so 330*16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # We need: for selected set S, for all i≠j in S: M[i,j] <= 2
    
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Multiple greedy with different orderings
    # Key insight: vectors that are "more compatible" with many others should be picked first
    # Compute for each vector, how many others it's compatible with (dot <= 2)
    
    # compat_count[i] = number of j ≠ i such that M[i,j] <= 2
    compat = (M <= 2).astype(np.int32)
    compat_count = compat.sum(axis=1)
    
    # Sort by descending compatibility count (most compatible first)
    order_comp = np.argsort(-compat_count)
    
    # Max-min diversity heuristic: pick first vector, then always pick the one 
    # that maximizes minimum distance to already-selected vectors
    def greedy_maxmin(start_idx, compat_mat, compat_cnt):
        """Greedy with max-min diversity after initial pick."""
        selected = [start_idx]
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[start_idx] = True
        
        # Precompute min distance to selected for each candidate
        min_dists = compat_mat.copy()  # will update
        
        for step in range(1, N):
            # Find candidate with max min-distance to selected
            # Only consider unselected candidates
            candidates_mask = ~selected_mask
            if not np.any(candidates_mask):
                break
            
            # For efficiency, track the minimum dot product violation count
            # Actually for max-min: we want to maximize the minimum "compatibility margin"
            # Use: among unselected, find one that has the largest minimum (2 - M[idx, sel])
            
            # Simple approach: for each unselected, compute min over selected of (2 - M[idx, sel])
            # This is expensive. Let's use a simpler heuristic:
            # Pick the unselected vector that is compatible with the most selected vectors
            # (i.e., has the fewest incompatibilities with current set)
            
            # Count incompatibilities with selected for each unselected
            # sub_M = M[np.ix_(np.where(candidates_mask)[0], np.where(selected_mask)[0])]
            # incomp = (sub_M > 2).sum(axis=1)
            # Pick argmin incomp, break ties by max compat_count
            
            # For speed, use vectorized approach
            sel_indices = np.where(selected_mask)[0]
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) == 0:
                break
            
            # Batch compute incompatibility counts
            # M[cand_indices][:, sel_indices] > 2, sum along axis 1
            sub = M[np.ix_(cand_indices, sel_indices)]
            incomp_counts = (sub > 2).sum(axis=1)
            
            # Pick the one with fewest incompatibilities
            min_incomp = np.min(incomp_counts)
            if min_incomp > 0:
                # No compatible candidate remains
                break
            
            # Among those with 0 incompatibilities, pick by highest compat_count
            zero_mask = incomp_counts == 0
            zero_cands = cand_indices[zero_mask]
            
            if len(zero_cands) == 0:
                break
            
            # Pick the one with highest original compat_count
            best_cand = zero_cands[np.argmax(compat_count[zero_cands])]
            
            selected.append(best_cand)
            selected_mask[best_cand] = True
        
        return selected
    
    orderings = [order_comp, np.arange(N)]
    # Add random permutations
    for _ in range(30):
        orderings.append(rng.permutation(N))
    
    # Run standard greedy on all orderings
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 512:
                break
    
    # Strategy 2: Max-min diversity greedy with different starting points
    # Try several different starting vectors
    start_candidates = list(order_comp[:10]) + list(rng.choice(N, size=20, replace=False))
    # Deduplicate
    start_candidates = list(dict.fromkeys(start_candidates))
    
    for start in start_candidates:
        if time.time() > deadline:
            break
        
        selected = greedy_maxmin(start, compat, compat_count)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 512:
                break
    
    # Local search: try to add vectors that are almost compatible
    if time.time() < deadline - 10 and best_size > 0:
        best_set_arr = np.array(best_set)
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set_arr] = True
        
        # Iteratively try to add vectors by removing blockers
        # For each unselected vector, find how many selected it conflicts with
        # If it conflicts with exactly 1, we can potentially remove that one and add this one
        # But that doesn't increase size. Instead, try 2-for-1 exchanges.
        
        # Better: try to find vectors incompatible with only 1 selected,
        # then try to find another vector that can replace the blocker
        
        unselected_indices = np.where(~in_set)[0]
        if len(unselected_indices) > 0:
            sub_M = M[np.ix_(unselected_indices, best_set_arr)]
            incomp_matrix = (sub_M > 2)  # shape (n_unsel, best_size)
            incomp_counts = incomp_matrix.sum(axis=1)
            
            # Vectors incompatible with exactly 1 selected
            one_incomp_mask = incomp_counts == 1
            one_incomp_indices = unselected_indices[one_incomp_mask]
            
            if len(one_incomp_indices) > 0 and time.time() < deadline - 10:
                # For each such vector j, the blocker is the unique selected i where M[j,i] > 2
                # Try to find another unselected vector k that is compatible with all selected 
                # except possibly i, and also compatible with j
                # Then we can remove i and add both j and k, net gain of 1
                
                # For efficiency, process in batches
                for j_idx in one_incomp_indices[:200]:  # limit to avoid timeout
                    if time.time() > deadline - 5:
                        break
                    
                    # Find the blocker
                    blocker_pos = np.where(incomp_matrix[j_idx])[0]
                    if len(blocker_pos) != 1:
                        continue
                    blocker_sel_idx = best_set_arr[blocker_pos[0]]
                    
                    # Find unselected vectors compatible with all selected except blocker
                    # i.e., incomp_counts' <= 1 and if 1, it's the blocker
                    # For simplicity: check all unselected against selected minus blocker
                    
                    sel_minus_blocker = np.array([s for s in best_set_arr if s != blocker_sel_idx])
                    if len(sel_minus_blocker) == 0:
                        continue
                    
                    # Check compatibility of unselected with sel_minus_blocker
                    sub2 = M[np.ix_(unselected_indices, sel_minus_blocker)]
                    incomp2 = (sub2 > 2).sum(axis=1)
                    
                    # Candidates: unselected, not j, compatible with sel_minus_blocker
                    cand_mask = incomp2 == 0
                    cand_mask[j_idx] = False  # exclude j itself
                    cands = unselected_indices[cand_mask]
                    
                    if len(cands) == 0:
                        continue
                    
                    # Check which cands are also compatible with j
                    # M[j, cands] <= 2
                    j_compat = M[j_idx, cands] <= 2
                    valid_cands = cands[j_compat]
                    
                    if len(valid_cands) > 0:
                        # We found a valid exchange: remove blocker, add j and one valid_cand
                        # This increases size by 1
                        new_set = list(best_set)
                        new_set.remove(blocker_sel_idx)
                        new_set.append(j_idx)
                        new_set.append(valid_cands[0])
                        
                        # Verify the new set is valid
                        new_set_arr = np.array(new_set)
                        sub_check = M[np.ix_(new_set_arr, new_set_arr)]
                        np.fill_diagonal(sub_check, 0)
                        if np.all(sub_check <= 2):
                            best_set = new_set
                            best_size = len(best_set)
                            best_set_arr = new_set_arr
                            in_set = np.zeros(N, dtype=bool)
                            in_set[best_set_arr] = True
                            unselected_indices = np.where(~in_set)[0]
                            if best_size >= 512:
                                break
    
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[best_set].astype(np.int64)
    
    # The construction enforces dot product <= 2 for vectors of norm sqrt(4).
    # Distance squared = 8 - 2*dot >= 8 - 4 = 4.
    # Max norm squared = 4.
    # Thus min distance >= max norm is guaranteed.
    
    return result
