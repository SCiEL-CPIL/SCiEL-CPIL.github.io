import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute pairwise difference matrices and squared norm differences
    # diff[i, j, :] = grid[j] - grid[i]
    # sq_diff[i, j] = sq_norms[j] - sq_norms[i]
    # Note: The determinant test uses rows (q - p, |q|^2 - |p|^2) where p is the "new" point and q are existing.
    # In our can_add logic, p is the candidate, q are in the set.
    # So we need rows (grid[q] - grid[p], sq_norms[q] - sq_norms[p]).
    
    # diff[i, j] = grid[j] - grid[i]
    diff = grid[np.newaxis, :, :] - grid[:, np.newaxis, :] 
    # sq_diff[i, j] = sq_norms[j] - sq_norms[i]
    sq_diff = sq_norms[np.newaxis, :] - sq_norms[:, np.newaxis]
    
    def can_add_to_set(points_indices, new_point_idx):
        """
        Check if new_point_idx can be added to the set of points given by indices in points_indices.
        Returns True if no 5 points (including new_point_idx) are cospherical/coplanar.
        """
        k = len(points_indices)
        if k < 4:
            return True
        
        # points_indices is a list/array of indices into grid.
        # new_point_idx is the index of the point we want to add.
        # We form a matrix for each 4-combination of points_indices.
        # Row r of the 4x4 matrix is [diff[new_point_idx, points_indices[r], 0], ..., diff[new_point_idx, points_indices[r], 2], sq_diff[new_point_idx, points_indices[r]]]
        
        # Get the relevant rows for the new point
        # shape: (k, 3) and (k,)
        d_vecs = diff[new_point_idx, points_indices, :]  # (k, 3)
        d_sqs = sq_diff[new_point_idx, points_indices]   # (k,)
        
        # We need to check all combinations of 4 indices from range(k)
        # Generate combinations of indices
        # For efficiency, if k is large, this is the bottleneck.
        # k is at most ~40-50. C(50, 4) = 230300.
        # We can vectorize the determinant calculation.
        
        combos = np.array(list(itertools.combinations(range(k), 4)), dtype=np.int32)
        if len(combos) == 0:
            return True
            
        # Extract rows
        # d_vecs[combos] -> (C, 4, 3)
        # d_sqs[combos] -> (C, 4)
        # Concatenate to (C, 4, 4)
        
        rows = np.concatenate([d_vecs[combos], d_sqs[combos][:, :, np.newaxis]], axis=2)
        
        # Calculate determinants
        # np.linalg.det supports batched inputs
        dets = np.linalg.det(rows)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_init(seed):
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected_indices = []
        for idx in order:
            if can_add_to_set(selected_indices, idx):
                selected_indices.append(idx)
        return selected_indices

    def local_search(initial_indices, max_time):
        current_indices = list(initial_indices)
        current_set = set(current_indices)
        best_indices = list(current_indices)
        best_size = len(current_indices)
        
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            # 1. Try to add any point
            added = False
            # Use a random order for candidates to avoid bias
            candidates = [i for i in range(N) if i not in current_set]
            np.random.shuffle(candidates)
            
            for idx in candidates:
                if can_add_to_set(current_indices, idx):
                    current_indices.append(idx)
                    current_set.add(idx)
                    added = True
                    if len(current_indices) > best_size:
                        best_indices = list(current_indices)
                        best_size = len(current_indices)
                    break # Restart scan to try adding more
            
            if added:
                continue
            
            # 2. No point could be added. Try 1-swap.
            swapped = False
            # Try several random swaps to escape local minima
            for _ in range(20):
                if time.time() - start_time > max_time:
                    break
                    
                out_idx = np.random.choice(current_indices)
                out_set = current_set - {out_idx}
                remaining_indices = [i for i in current_indices if i != out_idx]
                
                candidates = [i for i in range(N) if i not in out_set]
                np.random.shuffle(candidates)
                
                for in_idx in candidates[:50]: # Limit candidates to check for speed
                    if can_add_to_set(remaining_indices, in_idx):
                        # Swap is valid
                        current_indices = remaining_indices + [in_idx]
                        current_set = out_set | {in_idx}
                        swapped = True
                        break
                
                if swapped:
                    break
            
            if not swapped:
                # If no swap found, we might be stuck.
                # Try 2-swaps? (Remove 2, add 2)
                # This is more complex and expensive.
                # For now, if stuck, break and restart with new seed.
                # Or try removing 1 point and adding 2? (1-for-2 swap)
                # Let's try 1-for-2 swap.
                found_improvement = False
                for _ in range(10):
                    if time.time() - start_time > max_time:
                        break
                        
                    out_idx = np.random.choice(current_indices)
                    out_set = current_set - {out_idx}
                    remaining_indices = [i for i in current_indices if i != out_idx]
                    
                    # Try to add 2 points
                    candidates = [i for i in range(N) if i not in out_set]
                    np.random.shuffle(candidates)
                    
                    # Pick a random candidate to add first
                    for in_idx1 in candidates[:20]:
                        if can_add_to_set(remaining_indices, in_idx1):
                            temp_indices = remaining_indices + [in_idx1]
                            temp_set = out_set | {in_idx1}
                            
                            # Now try to add a second point
                            candidates2 = [i for i in range(N) if i not in temp_set]
                            np.random.shuffle(candidates2)
                            for in_idx2 in candidates2[:20]:
                                if can_add_to_set(temp_indices, in_idx2):
                                    # Valid 1-for-2 swap
                                    current_indices = temp_indices + [in_idx2]
                                    current_set = temp_set | {in_idx2}
                                    found_improvement = True
                                    if len(current_indices) > best_size:
                                        best_indices = list(current_indices)
                                        best_size = len(current_indices)
                                    break
                            if found_improvement:
                                break
                    if found_improvement:
                        break
                
                if not found_improvement:
                    break

        return best_indices

    # Run multiple restarts
    best_overall = []
    start_total = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    num_restarts = 0
    while time.time() - start_total < time_limit - safety_margin:
        num_restarts += 1
        seed = num_restarts
        init_indices = greedy_init(seed)
        
        elapsed = time.time() - start_total
        remaining = time_limit - safety_margin - elapsed
        if remaining < 10:
            break
            
        # Allocate time for local search
        # If we have a large set, we might want to spend more time polishing.
        # But local_search already includes a polish phase implicitly by trying to add.
        # Let's give it a fixed chunk of time, or proportional to remaining.
        ls_time = min(remaining * 0.7, 120.0) # Cap at 2 mins per restart?
        if ls_time < 5:
            break
            
        search_result = local_search(init_indices, ls_time)
        
        if len(search_result) > len(best_overall):
            best_overall = search_result
            
            # If we found a new best, we can try to polish it further with a deterministic pass
            # This is often faster and finds "easy" adds.
            current = list(best_overall)
            current_set = set(current)
            for idx in range(N):
                if idx not in current_set:
                    if can_add_to_set(current, idx):
                        current.append(idx)
                        current_set.add(idx)
                        
            if len(current) > len(best_overall):
                best_overall = current

    # Final polish on the absolute best
    if best_overall:
        current = list(best_overall)
        current_set = set(current)
        # Try adding in sorted order
        for idx in range(N):
            if idx not in current_set:
                if can_add_to_set(current, idx):
                    current.append(idx)
                    current_set.add(idx)
        best_overall = current

    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_overall].astype(np.int64)
