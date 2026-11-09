import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    deadline = time.time() + 590  # 10s safety margin
    np.random.seed(42)
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr (shape k,3) keeps the set sphere-free.
        Returns True if valid.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Precompute differences and squared norms
        p_sq = (p ** 2).sum(dtype=np.float64)
        arr_sq = (arr ** 2).sum(axis=1, dtype=np.float64)
        diff = (arr - p).astype(np.float64)
        sq_diff = arr_sq - p_sq
        
        # Check all 4-tuples
        for i0 in range(m - 3):
            for i1 in range(i0 + 1, m - 2):
                for i2 in range(i1 + 1, m - 1):
                    for i3 in range(i2 + 1, m):
                        mat = np.empty((4, 4), dtype=np.float64)
                        mat[0, 0:3] = diff[i0]
                        mat[0, 3] = sq_diff[i0]
                        mat[1, 0:3] = diff[i1]
                        mat[1, 3] = sq_diff[i1]
                        mat[2, 0:3] = diff[i2]
                        mat[2, 3] = sq_diff[i2]
                        mat[3, 0:3] = diff[i3]
                        mat[3, 3] = sq_diff[i3]
                        
                        d = np.linalg.det(mat)
                        if abs(d) < 0.5:
                            return False
        
        return True

    def greedy_build(order: np.ndarray, start_set: np.ndarray = None) -> np.ndarray:
        """Build a set by greedily adding points from order."""
        if start_set is None:
            curr_set = np.empty((0, 3), dtype=np.int64)
        else:
            curr_set = start_set.copy()
        
        for p in order:
            if time.time() > deadline:
                break
            if is_valid_addition(curr_set, p):
                curr_set = np.vstack([curr_set, p]) if len(curr_set) > 0 else p[None, :]
        
        return curr_set

    # --- Phase 1: Multiple Greedy Restarts ---
    best_set = np.empty((0, 3), dtype=np.int64)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        order = grid[np.random.permutation(num_points_total)]
        curr_set = greedy_build(order)
        
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # --- Phase 2: Local Search with Add and Swap ---
    current_set = best_set.copy()
    current_set_list = [tuple(p) for p in current_set]
    current_set_set = set(current_set_list)
    
    attempts = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try to add a point: check all remaining points
        remaining = [p for p in grid if tuple(p) not in current_set_set]
        for p in remaining:
            if time.time() > deadline:
                break
            if is_valid_addition(current_set, p):
                current_set = np.vstack([current_set, p])
                current_set_set.add(tuple(p))
                current_set_list.append(tuple(p))
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # 2. Try swaps: remove one point, add another
        if len(current_set) > 0 and len(remaining) > 0:
            # Try all pairs of (remove, add) - but that's O(k * (n-k)) which can be large
            # Sample instead
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                in_idx = np.random.randint(len(current_set))
                p_in = current_set[in_idx]
                
                out_idx = np.random.randint(len(remaining))
                p_out = remaining[out_idx]
                
                reduced_set = np.delete(current_set, in_idx, axis=0)
                
                if is_valid_addition(reduced_set, p_out):
                    current_set = reduced_set
                    current_set = np.vstack([current_set, p_out])
                    current_set_set.remove(tuple(p_in))
                    current_set_set.add(tuple(p_out))
                    current_set_list = [tuple(p) for p in current_set]
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
        
        attempts += 1

    return current_set.astype(np.int64)
