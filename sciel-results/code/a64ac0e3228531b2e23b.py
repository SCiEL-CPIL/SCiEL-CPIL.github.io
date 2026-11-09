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
        
        # Check all 4-combinations with early termination
        # For m up to ~80, C(m,4) is manageable with early exit
        if m <= 80:
            # Process in batches for memory efficiency
            batch_size = 2000
            combo_list = list(itertools.combinations(range(m), 4))
            total = len(combo_list)
            for start in range(0, total, batch_size):
                batch = combo_list[start:start+batch_size]
                if not batch:
                    break
                combos = np.array(batch, dtype=np.int64)
                sub_diff = diff[combos]
                sub_sq = sq_diff[combos]
                mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
                dets = np.linalg.det(mat)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True
        else:
            # For larger m, use a smarter approach: check against all 4-subsets
            # but with early termination and larger batches
            batch_size = 5000
            combo_list = list(itertools.combinations(range(m), 4))
            total = len(combo_list)
            for start in range(0, total, batch_size):
                batch = combo_list[start:start+batch_size]
                if not batch:
                    break
                combos = np.array(batch, dtype=np.int64)
                sub_diff = diff[combos]
                sub_sq = sq_diff[combos]
                mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
                dets = np.linalg.det(mat)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    def greedy_fill(order):
        selected = []
        for idx in order:
            if can_add_fast(np.array(selected), idx):
                selected.append(idx)
        return np.array(selected)
    
    def deterministic_polish(selected_arr):
        """Systematically try to add remaining points in sorted order."""
        current = list(selected_arr)
        current_set = set(current)
        for idx in range(N):
            if idx in current_set:
                continue
            if can_add_fast(np.array(current), idx):
                current.append(idx)
                current_set.add(idx)
        return np.array(current)

    def local_search(selected_arr, time_limit):
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add points in sorted order (deterministic first)
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
                for _ in range(10):
                    out_idx_pos = np.random.randint(len(current))
                    out_idx = current[out_idx_pos]
                    
                    candidates = [i for i in range(N) if i not in current_set]
                    if not candidates:
                        break
                    np.random.shuffle(candidates)
                    
                    swapped = False
                    for in_idx in candidates[:300]:  # Limit candidates to speed up
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
        
        # Apply deterministic polish
        current = deterministic_polish(current)
        
        # Give local search a chunk of time
        ls_deadline = time.time() + min(12, deadline - time.time())
        current = local_search(current, ls_deadline)
        
        # Apply deterministic polish again after local search
        current = deterministic_polish(current)
        
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
            
            # If we found a good solution, do more local search on it
            if time.time() < deadline - 30:
                extra_deadline = time.time() + min(15, deadline - time.time())
                current = local_search(best_arr, extra_deadline)
                current = deterministic_polish(current)
                if len(current) > len(best_arr):
                    best_arr = current.copy()
                    best_size = len(current)
    
    # Final deterministic polish on the best solution
    if len(best_arr) > 0 and time.time() < deadline - 5:
        best_arr = deterministic_polish(best_arr)
    
    return grid[best_arr]
