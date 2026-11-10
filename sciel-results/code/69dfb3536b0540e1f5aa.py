import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Find the maximum number of points in an 8x8x8 grid such that no 5 points 
    are cospherical.
    
    Returns:
        A numpy array of shape (k, 3) containing the coordinates of the points.
    """
    n = 8
    # Deadline guard: 630s limit, leave 30s safety margin
    deadline = time.time() + 570.0
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    
    # Precompute squared norms for faster calculation
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute all-pairs differences and squared norm differences for speed
    # diff[i,j] = grid[i] - grid[j]
    # sq_diff[i,j] = sq_norms[i] - sq_norms[j]
    diff = grid[:, np.newaxis, :] - grid[np.newaxis, :, :]  # shape (N, N, 3)
    sq_diff = sq_norms[:, np.newaxis] - sq_norms[np.newaxis, :]  # shape (N, N)
    
    def is_valid_addition_fast(S_indices, new_idx):
        """
        Check if adding new_idx to S_indices keeps the set valid.
        Uses precomputed diff and sq_diff for speed.
        """
        m = len(S_indices)
        if m < 4:
            return True
        
        p_idx = new_idx
        # Build W matrix: rows are [grid[q] - grid[p], sq_norms[q] - sq_norms[p]] for q in S
        # W has shape (m, 4)
        W = np.empty((m, 4), dtype=np.float64)
        # Vectorized: diff[p_idx, S_indices] gives (m, 3), sq_diff[p_idx, S_indices] gives (m,)
        W[:, 0:3] = diff[p_idx, S_indices].astype(np.float64)
        W[:, 3] = sq_diff[p_idx, S_indices].astype(np.float64)
        
        # Check if any 4 rows of W are linearly dependent
        # For speed, iterate through combinations but break early on failure
        for combo in itertools.combinations(range(m), 4):
            sub = W[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def is_valid_set(S_indices):
        """
        Full check if the set of indices is valid.
        """
        if len(S_indices) < 5:
            return True
        for combo in itertools.combinations(S_indices, 5):
            idxs = list(combo)
            p = idxs[0]
            # Check if p, q1, q2, q3, q4 are cospherical
            # Use the 4x4 determinant test with p as reference
            m = len(idxs) - 1  # 4 points
            W = np.empty((m, 4), dtype=np.float64)
            others = idxs[1:]
            W[:, 0:3] = diff[p, others].astype(np.float64)
            W[:, 3] = sq_diff[p, others].astype(np.float64)
            if abs(np.linalg.det(W)) < 0.5:
                return False
        return True

    # State: Set of selected indices
    best_set = set()
    best_size = 0
    best_list = []
    
    np.random.seed(42)
    
    # Phase 1: Multiple Greedy Strategies
    # Strategy 1: Random permutation
    # Strategy 2: Manhattan distance from center
    # Strategy 3: Coordinate sum ordering
    # Strategy 4: Max coordinate ordering
    # Strategy 5: Random with bias toward corners
    
    def greedy_with_order(order):
        """Run greedy addition with given order."""
        current_set = []
        current_set_set = set()
        for idx in order:
            if is_valid_addition_fast(current_set, idx):
                current_set.append(idx)
                current_set_set.add(idx)
        return current_set, current_set_set
    
    # Generate different orderings
    center = np.array([3.5, 3.5, 3.5])
    manhattan_dists = np.abs(grid - center).sum(axis=1)
    coord_sums = grid.sum(axis=1)
    max_coords = grid.max(axis=1)
    
    orders = []
    
    # Random orderings (multiple)
    for _ in range(8):
        orders.append(np.random.permutation(N))
    
    # Manhattan distance ordering (closest to center first, then farthest, etc.)
    orders.append(np.argsort(manhattan_dists))
    orders.append(np.argsort(-manhattan_dists))
    
    # Coordinate sum ordering
    orders.append(np.argsort(coord_sums))
    orders.append(np.argsort(-coord_sums))
    
    # Max coordinate ordering
    orders.append(np.argsort(max_coords))
    orders.append(np.argsort(-max_coords))
    
    # Corner-biased random: weight by distance from center
    corner_weights = 1.0 / (1.0 + manhattan_dists)
    for _ in range(4):
        probs = corner_weights / corner_weights.sum()
        order = np.random.choice(N, size=N, replace=False, p=probs)
        orders.append(order)
    
    # Run greedy for each ordering
    for order in orders:
        if time.time() > deadline - 200:
            break
        current_set, current_set_set = greedy_with_order(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set_set
            best_list = current_set
    
    # Phase 2: Local Search on the best set found
    def local_search(S_list, S_set, max_time=150.0):
        """
        Local search to improve the set.
        Tries:
        1. Adding points
        2. 1-for-1 swaps
        3. 1-for-2 swaps (remove 1, add 2)
        4. 2-for-1 swaps (remove 2, add 1)
        """
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # 1. Try to add any point
            for idx in range(N):
                if idx in S_set:
                    continue
                if is_valid_addition_fast(S_list, idx):
                    S_list.append(idx)
                    S_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swaps
            swapped = False
            for i, u in enumerate(S_list):
                if time.time() > deadline - 50:
                    return S_list, S_set
                for v in range(N):
                    if v in S_set:
                        continue
                    # Try replacing u with v
                    temp_list = [x for x in S_list if x != u]
                    if is_valid_addition_fast(temp_list, v):
                        S_list = temp_list + [v]
                        S_set = set(S_list)
                        swapped = True
                        improved = True
                        break
                if swapped:
                    break
            
            if improved:
                continue
            
            # 3. Try 1-for-2 swaps (remove 1, add 2) - can increase size
            found_12 = False
            for u in S_list:
                if time.time() > deadline - 50:
                    return S_list, S_set
                temp_list = [x for x in S_list if x != u]
                # Try adding two points to temp_list
                for v in range(N):
                    if v in S_set or v == u:
                        continue
                    if not is_valid_addition_fast(temp_list, v):
                        continue
                    temp_list2 = temp_list + [v]
                    for w in range(N):
                        if w in S_set or w == u or w == v:
                            continue
                        if is_valid_addition_fast(temp_list2, w):
                            S_list = temp_list2 + [w]
                            S_set = set(S_list)
                            found_12 = True
                            improved = True
                            break
                    if found_12:
                        break
                if found_12:
                    break
            
            if improved:
                continue
            
            # 4. Try 2-for-1 swaps (remove 2, add 1) - can enable future adds
            found_21 = False
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 50:
                        return S_list, S_set
                    u, v = S_list[i], S_list[j]
                    temp_list = [x for x in S_list if x != u and x != v]
                    for w in range(N):
                        if w in S_set or w == u or w == v:
                            continue
                        if is_valid_addition_fast(temp_list, w):
                            S_list = temp_list + [w]
                            S_set = set(S_list)
                            found_21 = True
                            improved = True
                            break
                    if found_21:
                        break
                if found_21:
                    break
            
        return S_list, S_set

    # Run local search on the best candidate
    if best_size > 0:
        best_list = list(best_set)
        best_list.sort()
        best_set = set(best_list)
        
        # Run local search with time limit
        final_list, final_set = local_search(best_list, best_set, max_time=180.0)
        
        # Verify final validity
        if is_valid_set(final_list):
            if len(final_list) > best_size:
                best_size = len(final_list)
                best_set = final_set
                best_list = final_list
        else:
            # Fallback: try to fix by removing problematic points
            # Keep the original best_set
            pass

    # Convert indices to coordinates
    result = grid[sorted(list(best_set))]
    
    # If no points found (shouldn't happen), return a small valid set
    if len(result) == 0:
        return grid[:4]

    return result.astype(np.int64)
