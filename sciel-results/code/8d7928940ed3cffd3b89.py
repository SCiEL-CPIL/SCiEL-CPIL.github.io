# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Find a large subset of the 8x8x8 grid with no five points co-spherical or co-planar.
    
    Uses a combination of:
    1. Multiple greedy initializations with different orderings
    2. Local search (swap/add/remove moves) to improve each solution
    3. Simulated annealing to escape local optima
    4. Deadline guard to return best valid solution before time limit
    """
    n = 8
    deadline = time.time() + 600  # Safety margin
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for each grid point
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)
    
    def compute_determinants_for_point(arr, p_idx):
        """
        For a candidate point p, compute determinants of all 4x4 matrices formed
        with 4 existing points. Returns array of |det| values.
        """
        m = len(arr)
        if m < 4:
            return np.array([1.0])
        
        p = grid[p_idx]
        # Build matrix: rows are (q - p, |q|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq_norms[arr] - sq_norms[p_idx]
        
        # Generate all combinations of 4 indices
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return np.array([1.0])
        
        sub = w[np.array(combos)]  # (C, 4, 4)
        return np.abs(np.linalg.det(sub))
    
    def is_valid(arr, p_idx):
        """Check if adding point p_idx to arr keeps the set valid."""
        dets = compute_determinants_for_point(arr, p_idx)
        return not np.any(dets < 0.5)
    
    def count_violations(arr):
        """
        Count the number of invalid 5-tuples (co-spherical or co-planar) in arr.
        Returns the minimum |det| across all 5-tuples (0 means valid).
        """
        m = len(arr)
        if m < 5:
            return 0.0
        
        min_det = float('inf')
        # For each point, check all 4-combinations of earlier points
        for i in range(4, m):
            p = arr[i]
            # Get indices of earlier points
            earlier = np.arange(i)
            if len(earlier) < 4:
                continue
            
            # Build matrix for point i relative to each earlier point
            w = np.empty((i, 4), dtype=np.float64)
            w[:, 0:3] = arr[:i] - p
            w[:, 3] = sq_norms[arr[:i]] - sq_norms[i]
            
            # Check all 4-combinations
            for combo in itertools.combinations(range(i), 4):
                sub = w[list(combo)]
                det_val = abs(np.linalg.det(sub))
                if det_val < min_det:
                    min_det = det_val
                    if min_det < 0.5:
                        return 0.0  # Found a violation
        
        return min_det
    
    def greedy_initialization(order):
        """Greedy: add points in given order if valid."""
        arr = np.empty((0, 3), dtype=np.int64)
        for p_idx in order:
            if is_valid(arr, p_idx):
                arr = np.vstack([arr, grid[p_idx]]) if len(arr) else grid[p_idx][None, :]
        return arr
    
    # Phase 1: Multiple greedy initializations
    best_arr = None
    best_size = 0
    
    np.random.seed(42)
    num_greedy = 20
    
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        order = np.random.permutation(num_points)
        arr = greedy_initialization(order)
        
        if len(arr) > best_size:
            best_size = len(arr)
            best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: return empty or a small valid set
        return np.array([], dtype=np.int64).reshape(0, 3)
    
    # Phase 2: Local search to improve best solution
    def local_search(arr, max_iters=1000):
        """
        Local search: try to add points, remove points, or swap points to improve.
        Returns improved array.
        """
        current = arr.copy()
        current_size = len(current)
        best_local = current.copy()
        best_local_size = current_size
        
        # Track which grid indices are in current set
        in_set = np.zeros(num_points, dtype=bool)
        for p in current:
            idx = np.where((grid == p).all(axis=1))[0]
            if len(idx) > 0:
                in_set[idx[0]] = True
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            improved = False
            
            # Try to add a point not in the set
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                # Try a random subset of candidates
                candidates = not_in_set[np.random.choice(len(not_in_set), 
                                                         min(50, len(not_in_set)), replace=False)]
                for p_idx in candidates:
                    if is_valid(current, p_idx):
                        current = np.vstack([current, grid[p_idx]])
                        in_set[p_idx] = True
                        improved = True
                        if len(current) > best_local_size:
                            best_local_size = len(current)
                            best_local = current.copy()
                        break
            
            # Try to swap: remove one point, add another
            if not improved and len(current) > 0:
                # Try removing a random point and adding a better one
                remove_idx = np.random.choice(len(current))
                removed_point = current[remove_idx]
                
                # Find the grid index of the removed point
                rem_grid_idx = np.where((grid == removed_point).all(axis=1))[0]
                if len(rem_grid_idx) == 0:
                    continue
                rem_grid_idx = rem_grid_idx[0]
                
                # Temporarily remove
                current_no_rem = np.delete(current, remove_idx, axis=0)
                in_set[rem_grid_idx] = False
                
                # Try to add a different point
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    candidates = not_in_set[np.random.choice(len(not_in_set),
                                                             min(30, len(not_in_set)), replace=False)]
                    for p_idx in candidates:
                        if is_valid(current_no_rem, p_idx):
                            # Check if this swap is an improvement (same size, but maybe enables future adds)
                            # For now, accept any valid swap
                            current = np.vstack([current_no_rem, grid[p_idx]])
                            in_set[rem_grid_idx] = False
                            in_set[p_idx] = True
                            improved = True
                            break
                
                if not improved:
                    # Restore
                    current = np.vstack([current, removed_point])
                    in_set[rem_grid_idx] = True
        
        return best_local
    
    # Apply local search
    best_arr = local_search(best_arr, max_iters=500)
    best_size = len(best_arr)
    
    # Phase 3: Simulated annealing
    def simulated_annealing(arr, num_iterations=2000, initial_temp=1.0, cooling_rate=0.999):
        """
        Simulated annealing: allow temporary violations to escape local optima.
        """
        current = arr.copy()
        current_size = len(current)
        best_sa = current.copy()
        best_sa_size = current_size
        
        # Track which grid indices are in current set
        in_set = np.zeros(num_points, dtype=bool)
        for p in current:
            idx = np.where((grid == p).all(axis=1))[0]
            if len(idx) > 0:
                in_set[idx[0]] = True
        
        temp = initial_temp
        
        for it in range(num_iterations):
            if time.time() > deadline - 30:
                break
            
            temp *= cooling_rate
            
            # Generate a move
            move_type = np.random.choice(['add', 'remove', 'swap'], p=[0.4, 0.2, 0.4])
            
            if move_type == 'add':
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                p_idx = not_in_set[np.random.choice(len(not_in_set))]
                
                # Check if valid
                if is_valid(current, p_idx):
                    new_current = np.vstack([current, grid[p_idx]])
                    in_set[p_idx] = True
                    current = new_current
                    if len(current) > best_sa_size:
                        best_sa_size = len(current)
                        best_sa = current.copy()
                else:
                    # Compute violation cost
                    dets = compute_determinants_for_point(current, p_idx)
                    min_det = np.min(dets)
                    cost = max(0, 0.5 - min_det)
                    
                    if np.random.random() < np.exp(-cost / max(temp, 1e-10)):
                        new_current = np.vstack([current, grid[p_idx]])
                        in_set[p_idx] = True
                        current = new_current
                        
            elif move_type == 'remove' and len(current) > 0:
                remove_idx = np.random.choice(len(current))
                removed_point = current[remove_idx]
                rem_grid_idx = np.where((grid == removed_point).all(axis=1))[0]
                if len(rem_grid_idx) == 0:
                    continue
                rem_grid_idx = rem_grid_idx[0]
                
                new_current = np.delete(current, remove_idx, axis=0)
                in_set[rem_grid_idx] = False
                current = new_current
                
            elif move_type == 'swap' and len(current) > 0:
                remove_idx = np.random.choice(len(current))
                removed_point = current[remove_idx]
                rem_grid_idx = np.where((grid == removed_point).all(axis=1))[0]
                if len(rem_grid_idx) == 0:
                    continue
                rem_grid_idx = rem_grid_idx[0]
                
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                p_idx = not_in_set[np.random.choice(len(not_in_set))]
                
                # Check if swap is valid
                current_no_rem = np.delete(current, remove_idx, axis=0)
                if is_valid(current_no_rem, p_idx):
                    new_current = np.vstack([current_no_rem, grid[p_idx]])
                    in_set[rem_grid_idx] = False
                    in_set[p_idx] = True
                    current = new_current
                    if len(current) > best_sa_size:
                        best_sa_size = len(current)
                        best_sa = current.copy()
        
        return best_sa
    
    # Run simulated annealing from best solution
    best_arr = simulated_annealing(best_arr, num_iterations=3000)
    best_size = len(best_arr)
    
    # Phase 4: Final local search polish
    best_arr = local_search(best_arr, max_iters=300)
    best_size = len(best_arr)
    
    # Phase 5: Try more greedy initializations if time permits
    if time.time() < deadline - 60:
        for i in range(20):
            if time.time() > deadline - 30:
                break
            
            order = np.random.permutation(num_points)
            arr = greedy_initialization(order)
            
            # Quick local search
            arr = local_search(arr, max_iters=100)
            
            if len(arr) > best_size:
                best_size = len(arr)
                best_arr = arr.copy()
    
    # Final validation
    def final_validate(arr):
        """Verify the solution is valid."""
        m = len(arr)
        if m < 5:
            return True
        
        for i in range(4, m):
            p = arr[i]
            earlier = np.arange(i)
            if len(earlier) < 4:
                continue
            
            w = np.empty((i, 4), dtype=np.float64)
            w[:, 0:3] = arr[:i] - p
            w[:, 3] = (arr[:i] ** 2).sum(axis=1) - (p ** 2).sum()
            
            for combo in itertools.combinations(range(i), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        
        return True
    
    # If final validation fails, fall back to a simpler valid set
    if not final_validate(best_arr):
        # Build a valid set greedily from scratch
        np.random.seed(0)
        order = np.random.permutation(num_points)
        arr = np.empty((0, 3), dtype=np.int64)
        for p_idx in order:
            if is_valid(arr, p_idx):
                arr = np.vstack([arr, grid[p_idx]]) if len(arr) else grid[p_idx][None, :]
        best_arr = arr
    
    return best_arr.astype(np.int64)


# EVOLVE-BLOCK-END
