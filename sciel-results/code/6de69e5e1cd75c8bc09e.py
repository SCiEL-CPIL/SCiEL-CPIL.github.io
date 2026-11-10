import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    # Cache for combination indices
    # Key: m (current set size)
    # Value: numpy array of shape (C(m,4), 4)
    combo_cache = {}

    def get_combos(m):
        if m in combo_cache:
            return combo_cache[m]
        
        # For very large m, generating all combos might be slow/memory heavy, 
        # but for m < 100, it's fine.
        if m > 100:
            # Fallback for safety, though unlikely to reach 100 in 580s with this check
            # Actually, if m is huge, the search will be very slow anyway.
            pass
            
        idx = np.arange(m)
        combos = list(itertools.combinations(idx, 4))
        if combos:
            arr = np.array(combos, dtype=np.int32)
        else:
            arr = np.empty((0, 4), dtype=np.int32)
        
        combo_cache[m] = arr
        return arr

    def can_add(arr_indices, p_idx):
        """
        Check if adding p_idx to the set defined by arr_indices is valid.
        arr_indices: array of indices of points in the set.
        p_idx: index of the new point.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        p = grid[p_idx]
        p_sq = grid_sq[p_idx]

        # Get points and squared norms of existing set
        S_pts = grid[arr_indices]
        S_sqs = grid_sq[arr_indices]

        # Compute differences: (m, 3)
        diffs = S_pts.astype(np.float64) - p.astype(np.float64)
        # Compute squared norm differences: (m,)
        sq_diffs = S_sqs - p_sq

        # Get combination indices
        combos = get_combos(m)
        
        if combos.shape[0] == 0:
            return True

        # Advanced indexing to create batch of 4x4 matrices
        # combos shape: (C, 4)
        # We want to index rows of diffs and sq_diffs
        # diffs[combos] will have shape (C, 4, 3)
        # sq_diffs[combos] will have shape (C, 4)
        
        # We need to form 4x4 matrices where row i is (diffs[i], sq_diffs[i])
        # Let's construct the matrices directly.
        
        # d_batch: (C, 4, 3)
        d_batch = diffs[combos]
        # s_batch: (C, 4)
        s_batch = sq_diffs[combos]
        
        # We need to stack them to get (C, 4, 4)
        # d_batch is (C, 4, 3). We want to add s_batch as the 4th column.
        # s_batch needs to be (C, 4, 1)
        s_batch_expanded = s_batch[:, :, np.newaxis]
        
        # Concatenate along the last axis
        # M_batch shape: (C, 4, 4)
        M_batch = np.concatenate([d_batch, s_batch_expanded], axis=-1)
        
        # Compute determinants
        dets = np.linalg.det(M_batch)
        
        # Check if any determinant is too small
        if np.any(np.abs(dets) < 0.5):
            return False
            
        return True

    def greedy_build(order):
        """
        Greedy build using the given order of indices.
        """
        curr_set = []
        curr_set_set = set()
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx in curr_set_set:
                continue
            
            if can_add(np.array(curr_set, dtype=np.int64), idx):
                curr_set.append(idx)
                curr_set_set.add(idx)
                
        return np.array(curr_set, dtype=np.int64)

    def local_search(initial_set):
        """
        Local search starting from initial_set.
        """
        current_set = initial_set.copy()
        current_set_set = set(current_set.tolist())
        
        # Complement: indices not in current_set
        all_indices = set(range(N))
        complement = list(all_indices - current_set_set)
        
        improved = True
        no_improve_count = 0
        max_no_improve = 50 # Limit local search duration per start
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Add
            # We iterate through the complement. 
            # To avoid O(N^2) behavior if we add many points, 
            # we can just scan the list. If we add a point, we restart the scan.
            # This is simple and effective.
            
            # Shuffle complement for randomness? 
            # Fixed order might be faster but less robust. 
            # Let's keep a stable order for speed, but maybe shuffle if stuck.
            # For now, iterate in existing order.
            
            added = False
            for idx in complement:
                if time.time() > deadline - 10:
                    return current_set
                
                # Check if idx is still in complement (it might have been added in a previous iteration of this loop if we didn't restart properly, 
                # but we break on add, so it's fine)
                
                if can_add(current_set, idx):
                    # Add it
                    current_set = np.append(current_set, idx)
                    current_set_set.add(idx)
                    complement.remove(idx)
                    improved = True
                    added = True
                    break # Restart the add phase
            
            if not added:
                # Phase 2: Swap
                # Try to swap one point out and one in
                # This is more expensive.
                # We try random swaps to save time.
                
                swaps_to_try = 20
                found_swap = False
                
                # Create a list of current indices for random selection
                curr_list = current_set.tolist()
                
                for _ in range(swaps_to_try):
                    if time.time() > deadline - 10:
                        return current_set
                    
                    # Pick random out
                    out_idx_val = curr_list[np.random.randint(len(curr_list))]
                    # Pick random in
                    in_idx_val = complement[np.random.randint(len(complement))]
                    
                    # Check if swap is valid
                    # Temporarily remove out_idx_val
                    # Find position of out_idx_val in current_set
                    out_pos = np.where(current_set == out_idx_val)[0][0]
                    
                    # Create reduced set
                    reduced_set = np.delete(current_set, out_pos)
                    
                    if can_add(reduced_set, in_idx_val):
                        # Perform swap
                        current_set[out_pos] = in_idx_val
                        current_set_set.remove(out_idx_val)
                        current_set_set.add(in_idx_val)
                        
                        # Update complement
                        complement.remove(in_idx_val)
                        complement.append(out_idx_val)
                        
                        improved = True
                        found_swap = True
                        break # Restart add phase after swap
                
                if not found_swap:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Kick: Remove a few random points to escape local optimum
                        if len(current_set) > 10:
                            kick_size = min(5, len(current_set) // 5)
                            kick_indices = np.random.choice(len(current_set), kick_size, replace=False)
                            for pos in kick_indices:
                                idx_to_remove = current_set[pos]
                                current_set_set.remove(idx_to_remove)
                                complement.append(idx_to_remove)
                            # Reconstruct current_set from set to maintain order? 
                            # Or just remove from array.
                            # Since we are in a loop, let's just mark this as improved 
                            # and let the add phase try to fill it back up.
                            # But we need to update the array.
                            # Easier: just break and restart local search with a slightly perturbed set?
                            # Or just return current_set and let the main loop do a new restart.
                            # Let's just break out of local search.
                            return current_set
                        else:
                            return current_set
                    no_improve_count = 0 # Reset if we did a kick? No, we returned.
                    # If we didn't kick, we are stuck.
                    # For simplicity, if we can't add or swap, we return.
                    # A more advanced search would do 2-for-1 moves, but that's complex.
                    # Given the time, simple add/swap is likely sufficient to beat 1.11.
                    return current_set
            else:
                no_improve_count = 0

        return current_set

    rng = np.random.RandomState(42)
    best_set = np.array([], dtype=np.int64)
    best_len = 0
    
    # Multi-start
    # We run multiple greedy builds, then local search on the best one found so far?
    # Or local search on each?
    # Local search is expensive. Let's run greedy, keep the best, then run local search on the best.
    # Or, run greedy + local search for a few restarts, keep the global best.
    
    num_restarts = 50 # We will break early based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        order = rng.permutation(N)
        greedy_set = greedy_build(order)
        
        if len(greedy_set) > best_len:
            best_set = greedy_set.copy()
            best_len = len(best_set)
        
        # Run local search on this greedy set
        # To save time, we might only run local search if the greedy set is promising
        # Or just run it on the current best?
        # Let's run local search on the result of this greedy build.
        
        ls_set = local_search(greedy_set)
        
        if len(ls_set) > best_len:
            best_set = ls_set.copy()
            best_len = len(best_set)

    return grid[best_set].astype(np.int64)
