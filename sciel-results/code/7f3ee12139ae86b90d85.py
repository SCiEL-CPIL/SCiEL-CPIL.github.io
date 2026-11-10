# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Construction: run multiple seeded random-order greedy passes over the grid,
    adding a point whenever it keeps the set sphere-free, then apply a local
    improvement (swap) pass. Keep the largest set found. Deterministic via
    np.random.seed(0). Includes a deadline guard to stay within 600s.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 580  # safety margin before 600s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if adding point p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def can_swap(arr: np.ndarray, remove_idx: int, p: np.ndarray) -> bool:
        """Check if removing arr[remove_idx] and adding p keeps the set sphere-free."""
        m = len(arr) - 1
        if m < 4:
            return True
        
        if remove_idx == 0:
            new_arr = arr[1:]
        elif remove_idx == m:
            new_arr = arr[:m]
        else:
            new_arr = np.concatenate([arr[:remove_idx], arr[remove_idx+1:]])
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = new_arr - p
        w[:, 3] = (new_arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Run a greedy pass over the given order of points."""
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        return arr

    def local_improve(arr: np.ndarray, time_budget: float) -> np.ndarray:
        """Try to improve arr by swapping one point for another."""
        arr = arr.copy()
        start_time = time.time()
        selected_set = set(map(tuple, arr.tolist()))
        
        # Precompute grid points not in selected set
        grid_not_selected = [p for p in grid if tuple(p) not in selected_set]
        
        max_candidates_per_swap = 100  # Limit candidates tried per swap attempt
        
        while time.time() - start_time < time_budget:
            improved = False
            # Shuffle the order we try removals to avoid bias
            indices = np.random.permutation(len(arr))
            for i in indices:
                if time.time() - start_time > time_budget:
                    break
                removed = arr[i]
                # Try a random subset of candidates
                if len(grid_not_selected) > max_candidates_per_swap:
                    candidate_indices = np.random.choice(len(grid_not_selected), 
                                                         size=max_candidates_per_swap, 
                                                         replace=False)
                else:
                    candidate_indices = np.random.permutation(len(grid_not_selected))
                
                found = False
                for ci in candidate_indices:
                    p = grid_not_selected[ci]
                    if can_swap(arr, i, p):
                        # Perform swap
                        new_arr = arr.copy()
                        new_arr[i] = p
                        arr = new_arr
                        selected_set.remove(tuple(removed))
                        selected_set.add(tuple(p))
                        # Remove p from candidates list
                        grid_not_selected[ci] = grid_not_selected[-1]
                        grid_not_selected.pop()
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not improved:
                break
        return arr

    np.random.seed(0)
    best = None
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = grid[np.random.permutation(len(grid))]
        arr = greedy_pass(order)
        
        if best is None or len(arr) > len(best):
            best = arr.copy()
    
    # Apply local improvement to the best solution
    if best is not None and time.time() < deadline:
        remaining = deadline - time.time() - 15
        if remaining > 5:
            improved = local_improve(best, remaining)
            if len(improved) > len(best):
                best = improved
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
