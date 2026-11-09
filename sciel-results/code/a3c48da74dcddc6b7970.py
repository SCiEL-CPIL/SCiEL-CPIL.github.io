# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from math import comb

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute: for every 5-tuple of grid points, determine if they are co-spherical/plane
    # A 5-tuple (i0,i1,i2,i3,i4) is "bad" if det of the 4x4 matrix with rows
    # (q_j - q_4, |q_j|^2 - |q_4|^2) for j=0..3 has |det| < 0.5
    # We'll precompute a set of frozensets of 5 indices that are bad.
    
    # To save memory, we can store bad 5-tuples as sorted tuples in a set.
    # Number of 5-tuples from 512 points: C(512,5) ~ 3.5e12 -- too many to enumerate all.
    # Instead, we'll compute on-the-fly but cache results.
    
    # Alternative: precompute for each pair of points, or use a different approach.
    
    # Let's think differently. The check for adding point p to existing set S is:
    # For every 4-subset {a,b,c,d} of S, check if {a,b,c,d,p} is co-spherical.
    # The determinant condition: form 4x4 matrix with rows (a-p, |a|^2-|p|^2), etc.
    # This is equivalent to checking if the 5 points lie on a common sphere/plane.
    
    # Key insight: 5 points are co-spherical iff the 5x5 matrix
    # [x y z x^2+y^2+z^2 1] has rank < 5, i.e., det = 0.
    # But the problem uses a specific 4x4 formulation. Let me re-read.
    
    # "form the 4x4 matrix with rows (q - p, |q|^2 - |p|^2)"
    # So rows are: (qx-px, qy-py, qz-pz, qx^2+qy^2+qz^2 - px^2-py^2-pz^2)
    # This is a 4x4 matrix. |det| < 0.5 means co-spherical (or degenerate).
    
    # For 5 points p, q1, q2, q3, q4, this checks if they're co-spherical.
    # Note: 4 points are always co-spherical (define a sphere), so we need 5.
    
    # Precomputation strategy: For each unordered 5-tuple of grid indices, 
    # compute whether it's bad. But C(512,5) is too large.
    
    # Instead: during greedy, when we have m points and try to add p,
    # we need to check C(m,4) combinations. For m=40, that's 91,390.
    # Each check is a 4x4 det. Vectorized, this is fast.
    
    # Let me optimize the vectorized check:
    # Given arr (m x 3) and p (3,), compute for all 4-combinations of arr:
    #   diff = arr[combo] - p  (4 x 3)
    #   sq_diff = grid_sq[arr[combo]] - grid_sq[p]  (4,)
    #   matrix = np.column_stack((diff, sq_diff))  (4 x 4)
    #   det = np.linalg.det(matrix)
    
    # Vectorized over all combos at once:
    #   combos: (C, 4) array of indices
    #   arr[combos]: (C, 4, 3)
    #   diff = arr[combos] - p[None, None, :]  (C, 4, 3)
    #   sq_diff = grid_sq[arr[combos]].ravel() - grid_sq[p]  (C*4,) -> reshape (C, 4)
    #   mat = np.concatenate([diff, sq_diff[:, :, None]], axis=2)  (C, 4, 4)
    #   dets = np.linalg.det(mat)  (C,)
    
    # This should be fast for C up to ~100k.
    
    def can_add_fast(arr: np.ndarray, p_idx: int, arr_sq: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        diff = arr - p  # (m, 3)
        sq_diff = arr_sq - p_sq  # (m,)
        
        # Generate all 4-combinations
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub_diff = diff[combos]  # (C, 4, 3)
        sub_sq = sq_diff[combos]  # (C, 4)
        mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)  # (C, 4, 4)
        dets = np.linalg.det(mat)  # (C,)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order_idx: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        arr_indices = []
        
        for idx in order_idx:
            if can_add_fast(arr, idx, arr_sq):
                p = grid[idx]
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, grid_sq[idx])
                arr_indices.append(idx)
                
        return arr

    # Local search: try to improve by swapping
    def local_search(arr: np.ndarray, max_iter: int = 50) -> np.ndarray:
        arr_sq = (arr ** 2).sum(axis=1)
        arr_indices = set()
        for p in arr:
            idx = np.where((grid == p).all(axis=1))[0][0]
            arr_indices.add(idx)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(len(arr)):
                # Try removing arr[i] and adding a new point
                remaining = np.delete(arr, i, axis=0)
                remaining_sq = np.delete(arr_sq, i)
                remaining_indices = arr_indices - {np.where((grid == arr[i]).all(axis=1))[0][0]}
                
                # Try adding each grid point not in remaining
                for new_idx in range(N):
                    if new_idx in remaining_indices:
                        continue
                    if can_add_fast(remaining, new_idx, remaining_sq):
                        new_arr = np.vstack([remaining, grid[new_idx]])
                        return new_arr  # Accept first improvement
            # If no swap found, try single removal + addition
        return arr

    best_arr = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 630
    safety_margin = 15
    deadline = start_time + time_limit - safety_margin
    
    rng = np.random.default_rng(42)
    
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        strategy = restart_count % 7
        
        if strategy == 0:
            order_idx = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N)
            order_idx = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N)
            order_idx = np.argsort(sums + noise)
        elif strategy == 3:
            noise = rng.random(N)
            order_idx = np.argsort(-sums + noise)
        elif strategy == 4:
            # Sort by distance from center, but reverse
            noise = rng.random(N)
            order_idx = np.argsort(-dists + noise)
        elif strategy == 5:
            # Random
            order_idx = rng.permutation(N)
        else:
            # Sort by x+y+z mod something
            key = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
            noise = rng.random(N)
            order_idx = np.argsort(key + noise)
            
        current_arr = greedy_fill(order_idx)
        
        # Try local search if we have time
        if time.time() < deadline - 30 and len(current_arr) >= 35:
            current_arr = local_search(current_arr, max_iter=20)
        
        if len(current_arr) > len(best_arr):
            best_arr = current_arr
            # Print progress for debugging
            # print(f"Restart {restart_count}: found {len(best_arr)} points")
            
        # If we've found a good solution, try to improve with more local search
        if len(best_arr) >= 40 and time.time() < deadline - 60:
            # Do more aggressive local search
            pass  # Could add more sophisticated moves
    
    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
