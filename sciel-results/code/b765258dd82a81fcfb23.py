import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute combinations of 4 indices for sets up to size MAX_M
    # We estimate we can reach up to 60-70 points. 
    # C(70, 4) ~ 916,895. C(80, 4) ~ 1,581,580.
    # Let's precompute up to 80 to be safe.
    MAX_M = 80
    all_combos = list(itertools.combinations(range(MAX_M), 4))
    combos_arr = np.array(all_combos, dtype=np.int32)
    num_combos = len(combos_arr)
    
    # Precompute cumulative counts of combinations to quickly slice
    # C(m, 4) for m=0..MAX_M
    # We can just use the fact that combinations are generated in lexicographic order
    # and C(m, 4) is the count for the first m elements.
    # Actually, itertools.combinations(range(MAX_M), 4) generates combos of 0..MAX_M-1.
    # If we have a set of size m, we need the first C(m, 4) combinations of the first m indices?
    # NO. The combinations in `combos_arr` are for indices 0..MAX_M-1.
    # If our set S has size m, we are using indices 0..m-1 (if we compact S).
    # So we need the combinations of 0..m-1.
    # The combinations of 0..m-1 are exactly the first C(m, 4) entries of `combos_arr` 
    # IF `combos_arr` was generated from `range(MAX_M)`.
    # Yes, `itertools.combinations` yields in lexicographic order.
    # The combinations of {0, ..., m-1} are a prefix of the combinations of {0, ..., MAX_M-1}.
    
    # Precompute C(m, 4) for m in 0..MAX_M
    c_m_4 = np.zeros(MAX_M + 1, dtype=np.int64)
    for m in range(4, MAX_M + 1):
        c_m_4[m] = int(np.math.comb(m, 4))
        
    def can_add_fast(arr_idx, p_idx):
        m = len(arr_idx)
        if m < 4:
            return True
        if m > MAX_M:
            # Fallback to slow check if we exceed precomputed limit
            # (Should not happen often if we cap search)
            return True 
            
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        # Get the relevant combinations
        k = c_m_4[m]
        if k == 0:
            return True
            
        # The indices in arr_idx are 0..m-1 (conceptually)
        # We need to map the combo indices (0..m-1) to the actual grid points in arr_idx
        # arr_idx is a list/array of indices into `grid`.
        # So grid[arr_idx] gives the points.
        # The combo indices refer to the position in the sorted/ordered list of S.
        # Since we maintain `arr_idx` as a list, and we use `itertools.combinations(range(m), 4)`,
        # the indices in combos_arr[0:k] are in 0..m-1.
        
        combos = combos_arr[:k]
        
        # Map to actual points
        # arr_pts = grid[arr_idx]
        # We need arr_pts[combos]
        # This is a fancy indexing operation.
        arr_pts = grid[arr_idx]
        sub_diff = arr_pts[combos] - p  # (k, 4, 3)
        
        arr_sq = grid_sq[arr_idx]
        sub_sq = arr_sq[combos] - p_sq  # (k, 4)
        
        mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2) # (k, 4, 4)
        
        # Check determinants
        # np.linalg.det can handle batches
        dets = np.linalg.det(mat)
        
        return bool(np.all(np.abs(dets) >= 0.5))

    def local_search(selected_arr, time_limit):
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        last_improvement = start_time
        
        # Pre-compute list of all points for random selection
        all_indices = list(range(N))
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any point
            # Shuffle all_indices for randomization
            np.random.shuffle(all_indices)
            for idx in all_indices:
                if idx in current_set:
                    continue
                if can_add_fast(np.array(current), idx):
                    current.append(idx)
                    current_set.add(idx)
                    improved = True
                    last_improvement = time.time()
                    break
            
            if improved:
                continue
            
            # 2. Try swaps: remove one, add another
            # Limit swap attempts per cycle to avoid getting stuck in local search loops
            swap_attempts = 0
            max_swap_attempts = 50
            
            while swap_attempts < max_swap_attempts:
                swap_attempts += 1
                
                # Pick a random point to remove
                out_pos = np.random.randint(len(current))
                out_idx = current[out_pos]
                
                # Pick a random candidate to add
                # We want a candidate NOT in current_set
                # Simple rejection sampling
                in_idx = np.random.choice(N)
                if in_idx in current_set:
                    continue
                
                # Test swap
                # Create temp list
                temp_current = current.copy()
                temp_current.pop(out_pos)
                
                if can_add_fast(np.array(temp_current), in_idx):
                    # Accept swap
                    current[out_pos] = in_idx
                    current_set.remove(out_idx)
                    current_set.add(in_idx)
                    improved = True
                    last_improvement = time.time()
                    break # Restart add phase
            
            if not improved:
                # Stuck. Try a "kick": remove 3 random points and try to refill
                # This helps escape local minima
                if len(current) > 10:
                    kick_size = 3
                    kick_indices = np.random.choice(len(current), size=kick_size, replace=False)
                    kick_indices = sorted(kick_indices, reverse=True) # Remove from end to keep indices stable
                    for pos in kick_indices:
                        idx = current.pop(pos)
                        current_set.remove(idx)
                    
                    # Try to refill
                    np.random.shuffle(all_indices)
                    for idx in all_indices:
                        if idx in current_set:
                            continue
                        if can_add_fast(np.array(current), idx):
                            current.append(idx)
                            current_set.add(idx)
                            if time.time() - last_improvement > 5:
                                # If we haven't improved in 5s, break kick
                                break
                    last_improvement = time.time()
                else:
                    # If we can't kick, break
                    break
        
        return np.array(current)

    best_size = 0
    best_arr = np.array([], dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 630 - 30  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    while time.time() < deadline:
        # Strategy 1: Random order
        # Strategy 2: Distance from center
        # Strategy 3: Sum of coords
        
        r = rng.random()
        if r < 0.33:
            order = rng.permutation(N)
        elif r < 0.66:
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(dists + noise)
        else:
            sums = grid.sum(axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(sums + noise)
            
        current = []
        for idx in order:
            if can_add_fast(np.array(current), idx):
                current.append(idx)
        
        # Local search
        # Allocate remaining time, but at least 10s per iteration
        ls_deadline = time.time() + min(20, deadline - time.time())
        current = local_search(current, ls_deadline)
        
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
            
            # If we found a new best, try to extend it further with a longer local search
            if time.time() < deadline - 30:
                extra_deadline = time.time() + min(30, deadline - time.time())
                current = local_search(best_arr, extra_deadline)
                if len(current) > len(best_arr):
                    best_arr = current.copy()
                    best_size = len(current)

    if len(best_arr) == 0:
        # Fallback: return empty or a small valid set
        return np.array([], dtype=np.int64)
        
    return grid[best_arr]
