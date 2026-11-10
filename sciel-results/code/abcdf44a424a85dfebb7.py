import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 580
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    n_grid = len(grid)
    norms_sq = (grid ** 2).sum(axis=1)
    
    def can_add_indices(arr_indices, p_idx):
        m = len(arr_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_norm_sq = norms_sq[p_idx]
        
        # Sort by distance to p to fail fast on close points
        dists = np.linalg.norm(grid[arr_indices] - p, axis=1)
        sorted_local_indices = np.argsort(dists)
        
        # Reorder w
        arr_coords = grid[arr_indices[sorted_local_indices]]
        arr_norms = norms_sq[arr_indices[sorted_local_indices]]
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, :3] = arr_coords - p
        w[:, 3] = arr_norms - p_norm_sq
        
        # Check all 4-combinations
        # If m is large, this is expensive.
        # But we expect to fail early if sorted by distance.
        
        for combo in itertools.combinations(range(m), 4):
            if np.abs(np.linalg.det(w[list(combo)])) < 0.5:
                return False
        return True

    def greedy_pass(rng):
        order = rng.permutation(n_grid)
        arr_indices = []
        for p_idx in order:
            if time.time() > deadline:
                break
            if can_add_indices(arr_indices, p_idx):
                arr_indices.append(p_idx)
        return arr_indices

    def local_improve(arr_indices, rng):
        arr_indices = list(arr_indices)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            m = len(arr_indices)
            
            # 1-Swap
            for i in range(m):
                if time.time() > deadline:
                    break
                arr_without_i = arr_indices[:i] + arr_indices[i+1:]
                # Try to add 1 or 2 points
                found = False
                # Try 1 point
                for _ in range(30):
                    p_idx = rng.randint(n_grid)
                    if p_idx in arr_without_i:
                        continue
                    if can_add_indices(arr_without_i, p_idx):
                        new_arr = arr_without_i + [p_idx]
                        # Try to extend with 1 more
                        for _ in range(10):
                            q_idx = rng.randint(n_grid)
                            if q_idx in new_arr:
                                continue
                            if can_add_indices(new_arr, q_idx):
                                new_arr.append(q_idx)
                                break
                        if len(new_arr) > m:
                            arr_indices = new_arr
                            improved = True
                            found = True
                            break
                if found:
                    break
            
            # 2-Swap (if 1-swap didn't improve)
            if not improved:
                for i in range(m):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, m):
                        if time.time() > deadline:
                            break
                        arr_without_ij = [x for k, x in enumerate(arr_indices) if k != i and k != j]
                        # Try to add 2 points
                        for _ in range(20):
                            p1 = rng.randint(n_grid)
                            p2 = rng.randint(n_grid)
                            if p1 == p2 or p1 in arr_without_ij or p2 in arr_without_ij:
                                continue
                            if can_add_indices(arr_without_ij, p1) and can_add_indices(arr_without_ij + [p1], p2):
                                new_arr = arr_without_ij + [p1, p2]
                                if len(new_arr) > m:
                                    arr_indices = new_arr
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
        return arr_indices

    np.random.seed(0)
    rng = np.random.RandomState(0)
    best = []
    
    # Run many restarts
    while time.time() < deadline - 5:
        arr = greedy_pass(rng)
        arr = local_improve(arr, rng)
        if len(arr) > len(best):
            best = arr
            
    if not best:
        return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.int64)
    else:
        return grid[np.array(best)].astype(np.int64)
