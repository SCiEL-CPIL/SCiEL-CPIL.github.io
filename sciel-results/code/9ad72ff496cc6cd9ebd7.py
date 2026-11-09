import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1)
    deadline = time.time() + 580  # Safety margin

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Returns True if valid, False otherwise.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # diff[i] = arr[i] - p
        diff = arr - p
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        sq_diff = sq_norms[np.where((grid == p).all(axis=1))[0][0]] - (p ** 2).sum()
        
        # Wait, p is a single point. We need its index in grid to get sq_norms.
        # Or just compute it directly.
        p_sq = (p ** 2).sum()
        sq_diff = sq_norms - p_sq
        
        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        # M[i] = [arr[i] - p, sq_norms[i] - p_sq]
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff
        
        # We need to check if any 4 rows are linearly dependent (det close to 0).
        # If m is small, we can check all combos.
        # If m is large, we can use a heuristic or early exit.
        # For m up to ~20, C(m,4) is manageable.
        # For larger m, we might need to be smarter.
        
        # Optimization: If m is large, we can't check all C(m,4).
        # However, the constraint is "no 5 cospherical".
        # If we add p, we need to ensure no 4 existing points + p are cospherical.
        # This is equivalent to: for any 4 existing points q1..q4, the 5 points are not cospherical.
        # The determinant test checks exactly this for the specific 4 points.
        
        # If m > 10, C(m,4) grows fast.
        # Let's implement a fast check for small m, and a randomized/heuristic check for large m?
        # Or just limit the search to sets where we can afford the check?
        # Actually, the score is proportional to k. We want max k.
        # But the check is O(C(m,4) * 4^3).
        # If m=20, C(20,4)=4845. 4845 * 64 ops ~ 300k ops. Very fast.
        # If m=50, C(50,4)=230300. 230300 * 64 ~ 15M ops. Still fast in NumPy if vectorized.
        
        if m <= 30:
            combos = list(itertools.combinations(range(m), 4))
            if not combos:
                return True
            combo_arr = np.array(combos, dtype=np.int32)
            M_sub = M[combo_arr]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
                return False
            return True
        else:
            # For large m, we can't check all combos efficiently in Python loops.
            # But we can use a property: if the points are in general position, det is large.
            # However, we must be strict.
            # Let's try to reduce the check.
            # If we have many points, it's likely invalid.
            # We can sample some combos to quickly reject.
            # But we need to be sure.
            # Actually, if we reach m=30, we've already passed 30 points.
            # The greedy might stop earlier.
            # Let's just do the full check if possible, or fail.
            # For now, let's allow up to m=40 with vectorized det.
            # C(40,4) = 91390.
            # Let's try.
            combos = list(itertools.combinations(range(m), 4))
            if not combos:
                return True
            combo_arr = np.array(combos, dtype=np.int32)
            M_sub = M[combo_arr]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
                return False
            return True

    def greedy_build(order):
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    best_set = np.empty((0, 3), dtype=np.int64)
    
    np.random.seed(42)
    
    # Phase 1: Multiple Greedy Restarts
    # Use different random orders to explore different parts of the grid.
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = np.random.permutation(num_points_total)
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search / Hill Climbing
    # Try to add more points to best_set.
    current_set = best_set.copy()
    current_set_set = set(map(tuple, current_set))
    remaining = [p for p in grid if tuple(p) not in current_set_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any remaining point
        for p in remaining:
            if is_valid_addition(current_set, p):
                current_set = np.vstack([current_set, p])
                current_set_set.add(tuple(p))
                remaining.remove(p)
                improved = True
                break
        
        if not improved:
            # Try swaps
            # For each point in current_set, remove it, try to add a point from remaining
            # This is O(N^2) per iteration, might be slow.
            # Let's just do a few random swaps.
            if len(current_set) > 0 and len(remaining) > 0:
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    idx_remove = np.random.randint(len(current_set))
                    p_remove = current_set[idx_remove]
                    # Remove p_remove
                    mask = np.ones(len(current_set), dtype=bool)
                    mask[idx_remove] = False
                    reduced_set = current_set[mask]
                    
                    # Try to add a random remaining point
                    idx_add = np.random.randint(len(remaining))
                    p_add = remaining[idx_add]
                    
                    if is_valid_addition(reduced_set, p_add):
                        # Perform swap
                        current_set = np.vstack([reduced_set, p_add])
                        current_set_set.remove(tuple(p_remove))
                        current_set_set.add(tuple(p_add))
                        remaining[idx_add] = p_remove
                        improved = True
                        break

    return current_set.astype(np.int64)
