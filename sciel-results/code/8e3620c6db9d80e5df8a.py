import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def can_add_fast(arr_idx, p_idx):
        m = len(arr_idx)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        arr_pts = grid[arr_idx]
        arr_sq = grid_sq[arr_idx]
        
        diff = arr_pts - p
        sq_diff = arr_sq - p_sq
        
        # For efficiency, check incrementally: 
        # We need to check all 4-combinations of the m existing points
        # But we can be smarter: if m is large, we can sample or use early termination
        
        # Generate combinations in batches to avoid huge memory usage
        # For m up to ~50, C(m,4) is manageable
        if m <= 50:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            if len(combos) == 0:
                return True
            sub_diff = diff[combos]
            sub_sq = sq_diff[combos]
            mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
            dets = np.linalg.det(mat)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # For larger m, sample combinations to save time
            # This is an approximation but should be fine for local search
            n_samples = min(10000, len(list(itertools.combinations(range(m), 4))))
            indices = np.random.choice(m, size=(n_samples, 4), replace=False)
            # Remove any rows with duplicates
            valid_mask = np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0, axis=1)
            indices = indices[valid_mask]
            if len(indices) == 0:
                return True
            sub_diff = diff[indices]
            sub_sq = sq_diff[indices]
            mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
            dets = np.linalg.det(mat)
            return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order):
        selected = []
        for idx in order:
            if can_add_fast(np.array(selected), idx):
                selected.append(idx)
        return np.array(selected)

    def local_search(selected_arr, time_limit):
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add points
            for idx in range(N):
                if idx in current_set:
                    continue
                if can_add_fast(np.array(current), idx):
                    current.append(idx)
                    current_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one point, try to add a different one
            if len(current) > 0:
                # Try multiple random removals before giving up
                for _ in range(5):
                    out_idx_pos = np.random.randint(len(current))
                    out_idx = current[out_idx_pos]
                    
                    candidates = [i for i in range(N) if i not in current_set]
                    np.random.shuffle(candidates)
                    
                    swapped = False
                    for in_idx in candidates[:200]:  # Limit candidates to speed up
                        temp_current = current.copy()
                        temp_current.pop(out_idx_pos)
                        if can_add_fast(np.array(temp_current), in_idx):
                            current[out_idx_pos] = in_idx
                            current_set.remove(out_idx)
                            current_set.add(in_idx)
                            swapped = True
                            improved = True
                            break
                    
                    if swapped:
                        break
                    elif not improved:
                        # Try another random removal
                        continue
            
            if not improved:
                # No improvement possible, break
                break

        return np.array(current)

    best_size = 0
    best_arr = np.array([], dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 630 - 20  # Safety margin
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        r = rng.random()
        if r < 0.3:
            order = rng.permutation(N)
        elif r < 0.6:
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(dists + noise)
        else:
            sums = grid.sum(axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(sums + noise)
            
        current = greedy_fill(order)
        
        # Give local search a chunk of time
        ls_deadline = time.time() + min(15, deadline - time.time())
        current = local_search(current, ls_deadline)
        
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
            
            # If we found a good solution, do more local search on it
            if time.time() < deadline - 30:
                extra_deadline = time.time() + min(20, deadline - time.time())
                current = local_search(best_arr, extra_deadline)
                if len(current) > len(best_arr):
                    best_arr = current.copy()
                    best_size = len(current)
    
    return grid[best_arr]
