import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def can_add(p_idx, S_indices, S_points):
        m = len(S_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        # W: (m, 4)
        # W[i] = (S_points[i] - p, S_sq[i] - p_sq)
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = S_points - p
        W[:, 3] = (grid_sq[S_indices] - p_sq)
        
        if m < 4: return True
        
        # We check all combinations of 4 indices from range(m)
        # If m is large, we process in batches to manage memory.
        
        # Estimate number of combinations
        # C(m, 4) = m*(m-1)*(m-2)*(m-3)/24
        if m <= 80:
            combos = list(itertools.combinations(range(m), 4))
            if not combos: return True
            idx_array = np.array(combos, dtype=np.int32)
            sub = W[idx_array]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Batch processing for large m
            batch_size = 50000
            batch = []
            for combo in itertools.combinations(range(m), 4):
                batch.append(combo)
                if len(batch) >= batch_size:
                    idx_array = np.array(batch, dtype=np.int32)
                    sub = W[idx_array]
                    dets = np.linalg.det(sub)
                    if np.any(np.abs(dets) < 0.5):
                        return False
                    batch = []
            if batch:
                idx_array = np.array(batch, dtype=np.int32)
                sub = W[idx_array]
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 550 # Leave some margin
    
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Init
        current_indices = []
        current_points = np.empty((0, 3), dtype=np.int64)
        
        # Random order for init
        order = np.random.permutation(len(grid))
        for idx in order:
            if time.time() - start_time > time_limit: break
            if can_add(idx, current_indices, current_points):
                current_indices.append(idx)
                current_points = np.vstack([current_points, grid[idx]]) if len(current_points) else grid[idx][None, :]
                
                if len(current_indices) > best_size:
                    best_size = len(current_indices)
                    best_set = list(current_indices)
        
        # Local Search
        while time.time() - start_time < time_limit:
            added = False
            # Try to add a point
            # Shuffle candidates to avoid bias
            candidates = np.random.permutation(len(grid))
            
            # Optimization: Only try candidates not already in set
            # Using a set for O(1) lookup
            in_set = set(current_indices)
            
            for idx in candidates:
                if idx in in_set:
                    continue
                if can_add(idx, current_indices, current_points):
                    # Add it
                    current_indices.append(idx)
                    current_points = np.vstack([current_points, grid[idx]])
                    in_set.add(idx)
                    added = True
                    if len(current_indices) > best_size:
                        best_size = len(current_indices)
                        best_set = list(current_indices)
                    break
            
            if added:
                continue
            
            # If no point could be added, we are at a local maximum.
            # Perturb: remove k points
            m = len(current_indices)
            if m == 0:
                break
            
            # Randomly remove 1 to 3 points
            k = np.random.randint(1, 4)
            if k > m: k = m
            
            remove_indices = np.random.choice(m, k, replace=False)
            # Remove from end first to keep list stable? Or just filter.
            # Filter is O(m).
            new_indices = []
            new_points_list = []
            for i in range(m):
                if i not in remove_indices:
                    new_indices.append(current_indices[i])
                    new_points_list.append(current_points[i])
            
            current_indices = new_indices
            current_points = np.array(new_points_list) if new_points_list else np.empty((0,3), dtype=np.int64)
            
            # Update best (in case we removed points but somehow size increased? No, but maybe previous state was better? 
            # Actually, we update best on add. But if we start with a good state and remove, we might drop below best.
            # The best_set is kept globally.
            pass

    # Convert best_set to array
    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_set].astype(np.int64)
