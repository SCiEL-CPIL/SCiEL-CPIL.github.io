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
        
        # Batch processing for all combinations
        # To allow early exit, we process in small chunks
        chunk_size = 10000
        total_combos = 0
        
        for combo in itertools.combinations(range(m), 4):
            # We can't easily vectorize a single combo without creating an array.
            # But creating an array for each combo is slow.
            # Better to stick to the batch approach but check early.
            pass
        
        # Revert to batch approach but with early exit logic on the batch result
        # We generate combinations in chunks
        combos = []
        for c in itertools.combinations(range(m), 4):
            combos.append(c)
            if len(combos) >= chunk_size:
                idx_array = np.array(combos, dtype=np.int32)
                sub = W[idx_array]
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
                combos = []
        
        if combos:
            idx_array = np.array(combos, dtype=np.int32)
            sub = W[idx_array]
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        
        return True

    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 550
    
    np.random.seed(42)
    
    # Phase 1: Multiple Greedy Inits
    # We run several greedy inits to find a good starting point
    init_count = 20
    best_init_set = []
    best_init_size = 0
    
    for i in range(init_count):
        if time.time() - start_time > time_limit * 0.2: # Use 20% of time for init
            break
        current_indices = []
        current_points = np.empty((0, 3), dtype=np.int64)
        order = np.random.permutation(len(grid))
        for idx in order:
            if can_add(idx, current_indices, current_points):
                current_indices.append(idx)
                current_points = np.vstack([current_points, grid[idx]]) if len(current_points) else grid[idx][None, :]
        
        if len(current_indices) > best_init_size:
            best_init_size = len(current_indices)
            best_init_set = list(current_indices)
            best_init_points = current_points.copy()
            
    # Use the best init for local search
    if not best_init_set:
         # Fallback to single greedy if no time or failed
         current_indices = []
         current_points = np.empty((0, 3), dtype=np.int64)
         order = np.random.permutation(len(grid))
         for idx in order:
             if can_add(idx, current_indices, current_points):
                 current_indices.append(idx)
                 current_points = np.vstack([current_points, grid[idx]]) if len(current_points) else grid[idx][None, :]
         best_init_set = current_indices
         best_init_points = current_points

    current_indices = best_init_set
    current_points = best_init_points
    best_size = len(current_indices)
    best_set = list(current_indices)

    # Phase 2: Local Search
    while time.time() - start_time < time_limit:
        added = False
        candidates = np.random.permutation(len(grid))
        in_set = set(current_indices)
        
        for idx in candidates:
            if idx in in_set:
                continue
            if can_add(idx, current_indices, current_points):
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
        
        # Stuck: Perturb
        m = len(current_indices)
        if m == 0:
            break
        
        # Remove 1 to 5 points randomly
        k = np.random.randint(1, 6)
        if k > m: k = m
        
        remove_indices = np.random.choice(m, k, replace=False)
        remove_set = set(remove_indices)
        
        new_indices = []
        new_points_list = []
        for i in range(m):
            if i not in remove_set:
                new_indices.append(current_indices[i])
                new_points_list.append(current_points[i])
        
        current_indices = new_indices
        current_points = np.array(new_points_list) if new_points_list else np.empty((0,3), dtype=np.int64)

    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_set].astype(np.int64)
