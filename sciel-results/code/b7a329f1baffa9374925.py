# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical.
    Uses multi-start greedy + local search (swap moves) with time budget.
    """
    n = 8
    deadline = time.time() + 580  # 20s safety margin
    
    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point
    
    # Precompute index lookup
    idx_map = {}
    for i, p in enumerate(grid):
        idx_map[tuple(p)] = i
    
    def is_valid_set(indices: np.ndarray) -> bool:
        """Check if the set of grid indices has no 5 co-spherical points."""
        m = len(indices)
        if m < 5:
            return True
        pts = grid[indices]
        sqs = sq[indices]
        for i in range(m):
            p = pts[i]
            pi_sq = sqs[i]
            others = np.delete(indices, i)
            if len(others) < 4:
                continue
            w = np.empty((len(others), 4), dtype=np.float64)
            w[:, 0:3] = grid[others] - p
            w[:, 3] = sq[others] - pi_sq
            for combo in itertools.combinations(range(len(others)), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True
    
    def can_add_to_set(arr_indices: np.ndarray, new_idx: int) -> bool:
        """Check if adding new_idx to the set keeps it valid.
        The new point must not form a co-spherical 5-tuple with any 4 existing points."""
        m = len(arr_indices)
        if m < 4:
            return True
        p = grid[new_idx]
        pi_sq = sq[new_idx]
        others = arr_indices
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = grid[others] - p
        w[:, 3] = sq[others] - pi_sq
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True
    
    def greedy_build(order: np.ndarray, start_arr: np.ndarray = None) -> np.ndarray:
        """Greedy: add points in given order if feasible."""
        if start_arr is None:
            arr = np.empty(0, dtype=np.int64)
        else:
            arr = start_arr.copy()
        arr_set = set(arr.tolist())
        for idx in order:
            if idx in arr_set:
                continue
            if can_add_to_set(arr, idx):
                arr = np.append(arr, idx)
                arr_set.add(idx)
        return arr
    
    def local_search(arr: np.ndarray, max_swaps: int = 200) -> np.ndarray:
        """Try swap moves: remove one, add another, keep if valid and same size or larger."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline:
            improved = False
            # Try each removal
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without = np.delete(current, i)
                # Try adding each complement point
                for j in range(len(comp)):
                    if time.time() > deadline:
                        break
                    add_idx = comp[j]
                    if can_add_to_set(without, add_idx):
                        # Valid swap
                        new_arr = np.concatenate([without, [add_idx]])
                        current = new_arr
                        current_set = set(current.tolist())
                        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                        improved = True
                        swaps_done += 1
                        break  # Restart inner loop with new complement
                if improved:
                    break
        return current
    
    def local_search_2opt(arr: np.ndarray, max_swaps: int = 50) -> np.ndarray:
        """Try 2-opt moves: swap two points at once."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline:
            improved = False
            # Try each pair of removals
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current)):
                    if time.time() > deadline:
                        break
                    removed_i = current[i]
                    removed_j = current[j]
                    # Remove both
                    mask = np.ones(len(current), dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    without = current[mask]
                    # Try adding each pair of complement points
                    for a in range(len(comp)):
                        if time.time() > deadline:
                            break
                        for b in range(a+1, len(comp)):
                            if time.time() > deadline:
                                break
                            add_a = comp[a]
                            add_b = comp[b]
                            # Check if adding both keeps valid
                            new_arr = np.concatenate([without, [add_a, add_b]])
                            if is_valid_set(new_arr):
                                current = new_arr
                                current_set = set(current.tolist())
                                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                                improved = True
                                swaps_done += 1
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return current
    
    def local_search_add(arr: np.ndarray) -> np.ndarray:
        """Try to add more points after local search."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        for idx in comp:
            if time.time() > deadline:
                break
            if can_add_to_set(current, idx):
                current = np.append(current, idx)
                current_set.add(idx)
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        return current
    
    rng = np.random.RandomState(42)
    best = None
    best_len = 0
    
    # Phase 1: Multiple greedy restarts
    while time.time() < deadline - 60:  # Leave 60s for local search
        order = rng.permutation(N)
        arr = greedy_build(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
    
    # Phase 2: Local search on best
    if best is not None and time.time() < deadline:
        best = local_search(best)
        best = local_search_add(best)
        best_len = len(best)
    
    # Phase 3: 2-opt local search
    if best is not None and time.time() < deadline - 30:
        best = local_search_2opt(best, max_swaps=30)
        best = local_search_add(best)
        best_len = len(best)
    
    # Phase 4: More restarts with local search if time permits
    while time.time() < deadline - 30:
        order = rng.permutation(N)
        arr = greedy_build(order)
        arr = local_search(arr, max_swaps=50)
        arr = local_search_add(arr)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
    
    if best is None:
        # Fallback: just first 4 points
        best = np.array([0, 1, 2, 3], dtype=np.int64)
    
    return grid[best].astype(np.int64)


# EVOLVE-BLOCK-END
