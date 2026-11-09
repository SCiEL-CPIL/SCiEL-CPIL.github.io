# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def can_add_fast(arr: np.ndarray, p_idx: int, arr_sq: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        diff = arr - p  # (m, 3)
        sq_diff = arr_sq - p_sq  # (m,)
        
        # Generate all 4-combinations
        if m < 4:
             return True
             
        # We need to check if any 4 points from arr are cospherical with p.
        # This is equivalent to checking if the determinant of the matrix 
        # formed by (q_i - p, |q_i|^2 - |p|^2) is 0 for any subset of 4 q's.
        
        # To speed this up, we can check all combinations.
        # For m ~ 40, C(40, 4) = 91,390. This is manageable.
        # For m ~ 50, C(50, 4) = 230,300. Still manageable with numpy.
        
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub_diff = diff[combos]  # (C, 4, 3)
        sub_sq = sq_diff[combos]  # (C, 4)
        mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)  # (C, 4, 4)
        dets = np.linalg.det(mat)  # (C,)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order_idx: np.ndarray, existing: np.ndarray = None) -> np.ndarray:
        if existing is None:
            arr = np.empty((0, 3), dtype=np.int64)
            arr_sq = np.empty(0, dtype=np.int64)
            arr_set = set()
        else:
            arr = existing.copy()
            arr_sq = (arr ** 2).sum(axis=1)
            arr_set = set(tuple(p) for p in arr)
            
        # We want to add points from order_idx that are not already in arr
        # But order_idx is a permutation of all N points.
        # We should only consider points not in arr_set.
        
        for idx in order_idx:
            p_tuple = tuple(grid[idx])
            if p_tuple in arr_set:
                continue
                
            if can_add_fast(arr, idx, arr_sq):
                p = grid[idx]
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.int64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                arr_set.add(p_tuple)
                
        return arr

    best_arr = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    deadline = start_time + 600 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Pre-compute some sorting keys
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    prods = grid[:, 0] * grid[:, 1] * grid[:, 2]
    key1 = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
    maxes = grid.max(axis=1)
    
    # Strategies for ordering
    strategies = [
        lambda: rng.permutation(N),
        lambda: np.argsort(dists + rng.random(N)),
        lambda: np.argsort(sums + rng.random(N)),
        lambda: np.argsort(-sums + rng.random(N)),
        lambda: np.argsort(-dists + rng.random(N)),
        lambda: np.argsort(key1 + rng.random(N)),
        lambda: np.argsort(prods + rng.random(N)),
        lambda: np.argsort(maxes + rng.random(N)),
    ]
    
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        
        # 1. Try a new greedy start
        if iteration % 3 == 1 or len(best_arr) == 0:
            strat_idx = iteration % len(strategies)
            order_idx = strategies[strat_idx]()
            current_arr = greedy_fill(order_idx)
            
            if len(current_arr) > len(best_arr):
                best_arr = current_arr.copy()
        
        # 2. Iterated Local Search on best_arr
        # Remove k random points and re-greedy fill
        if len(best_arr) > 0 and time.time() < deadline - 5:
            k = 2 + (iteration % 3) # Remove 2, 3, or 4 points
            if len(best_arr) > k:
                remove_indices = rng.choice(len(best_arr), size=k, replace=False)
                # Create a mask of points to keep
                keep_mask = np.ones(len(best_arr), dtype=bool)
                keep_mask[remove_indices] = False
                remaining_arr = best_arr[keep_mask]
                
                # Create an order that prioritizes the removed points first, then others
                # This encourages re-inserting them if they fit, or finding better replacements
                removed_tuples = [tuple(p) for p in best_arr[remove_indices]]
                
                # Build order: removed points (shuffled) + others (shuffled)
                all_indices = np.arange(N)
                # We need to map tuples to indices? 
                # It's easier to just shuffle all N points, but put removed ones at the front with high probability.
                
                # Simple approach: Shuffle all, but ensure removed points are considered early.
                # Actually, greedy_fill skips points already in `existing`.
                # So we pass `remaining_arr` as existing.
                # The order should prioritize points that were removed.
                
                # Let's create a custom order
                # Indices of removed points
                rem_idx_list = [np.where((grid == np.array(rt)).all(axis=1))[0][0] for rt in removed_tuples]
                
                # Shuffle the rest
                other_indices = [i for i in range(N) if i not in rem_idx_list]
                rng.shuffle(other_indices)
                
                # Order: shuffled removed + shuffled others
                order_idx = np.array(rem_idx_list + other_indices)
                
                new_arr = greedy_fill(order_idx, existing=remaining_arr)
                
                if len(new_arr) > len(best_arr):
                    best_arr = new_arr.copy()
                    
    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
