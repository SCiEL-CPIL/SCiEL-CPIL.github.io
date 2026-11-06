"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
import sys

N = 64
NUM_POINTS = N * N

# EVOLVE-BLOCK-START

def construct():
    """
    Find a large set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    Uses vectorized NumPy compatibility checks with simulated annealing and remove-1-add-2 moves.
    """
    deadline = time.time() + 280  # Leave margin before 300s limit
    
    # Pre-compute all points
    points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Pre-compute squared distances between all pairs
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy  # (NUM_POINTS, NUM_POINTS)
    
    # For each point p, precompute a dictionary: distance -> list of points at that distance
    # This allows fast lookup for isosceles triangle checks
    # dist_map[p][d] = set of indices q such that D2[p, q] == d
    dist_map = [None] * NUM_POINTS
    for p in range(NUM_POINTS):
        row = D2[p, :]
        unique_d, inverse = np.unique(row, return_inverse=True)
        dmap = {}
        for d_idx, d_val in enumerate(unique_d):
            if d_val == 0:
                continue  # Skip self-distance
            pts_at_d = np.where(inverse == d_idx)[0]
            dmap[int(d_val)] = pts_at_d
        dist_map[p] = dmap
    
    # For each pair (i, j), precompute the set of points k that are equidistant from i and j
    # i.e., D2[i, k] == D2[j, k]
    # This is the perpendicular bisector condition
    # We'll compute this on-the-fly for efficiency in memory
    
    def is_compatible_vectorized(selected, p):
        """
        Check if adding point p to the selected set creates any isosceles triangle.
        Returns True if p can be added without creating an isosceles triangle.
        Uses vectorized NumPy operations for speed.
        """
        if selected[p]:
            return False
        
        idx = np.where(selected)[0]
        K = len(idx)
        
        if K < 2:
            return True
        
        # Get distances from p to all selected points
        dp = D2[p, idx]  # shape (K,)
        
        # Get pairwise distances among selected points
        D2_sel = D2[np.ix_(idx, idx)]  # shape (K, K)
        
        # Check 1: For each i in idx, is there a j in idx (j != i) such that D2[i, p] == D2[i, j]?
        # This means vertex at i: dist(i,p) == dist(i,j)
        # For each i, check if dp[i] appears in D2_sel[i, :] excluding position i
        # Vectorized: for each row i of D2_sel, check if dp[i] is present in that row (excluding self)
        
        # Create a mask: for each (i, j) pair, check if D2_sel[i, j] == dp[i]
        # and j != i (handled by excluding diagonal)
        # Also need j != p, but p is not in idx so this is automatic
        
        # For vertex at i: exists j in idx, j != i, such that D2[i,j] == D2[i,p]
        # D2[i,p] = dp[i]
        # So check: for each i, is there j != i with D2_sel[i,j] == dp[i]?
        
        # Vectorized approach:
        # match1[i, j] = (D2_sel[i, j] == dp[i]) and (j != i)
        dp_col = dp[:, None]  # shape (K, 1)
        match1 = (D2_sel == dp_col)  # shape (K, K): match1[i,j] = (D2_sel[i,j] == dp[i])
        # Exclude diagonal
        match1[np.diag_indices(K)] = False
        if np.any(match1):
            return False
        
        # Check 2: For each j in idx, is there an i in idx (i != j) such that D2[j, p] == D2[i, j]?
        # This means vertex at j: dist(j,p) == dist(i,j)
        # D2[j,p] = dp[j]
        # So check: for each j, is there i != j with D2_sel[i,j] == dp[j]?
        # Note: D2_sel[i,j] = D2[idx[i], idx[j]]
        
        dp_row = dp[None, :]  # shape (1, K)
        match2 = (D2_sel == dp_row)  # shape (K, K): match2[i,j] = (D2_sel[i,j] == dp[j])
        match2[np.diag_indices(K)] = False
        if np.any(match2):
            return False
        
        # Check 3: Is there a k in idx such that D2[i,k] == D2[j,k] for some i != j?
        # Wait, this is: for the triple (i, j, p), check if dist(i,p) == dist(j,p)
        # i.e., dp[i] == dp[j] for some i != j
        # This means p is equidistant from i and j, so vertex at p
        
        # Check if any two values in dp are equal
        if K >= 2:
            # Check for duplicate distances from p to selected points
            unique_dp = np.unique(dp)
            if len(unique_dp) < K:
                return False
        
        return True
    
    def count_conflicts_fast(selected):
        """Count isosceles triangles in the selected set using vectorized operations."""
        idx = np.where(selected)[0]
        K = len(idx)
        if K < 3:
            return 0
        
        D2_sel = D2[np.ix_(idx, idx)]
        
        # For each pair (i, j), check if there exists k != i, j such that:
        # 1. D2_sel[i,k] == D2_sel[i,j] (vertex at i)
        # 2. D2_sel[j,k] == D2_sel[i,j] (vertex at j)  
        # 3. D2_sel[i,k] == D2_sel[j,k] (vertex at k)
        
        conflicts = 0
        for i in range(K):
            for j in range(i + 1, K):
                dij2 = D2_sel[i, j]
                
                # Check 1: exists k != i,j with D2_sel[i,k] == dij2
                row_i = D2_sel[i, :].copy()
                row_i[i] = -1
                row_i[j] = -1
                if np.any(row_i == dij2):
                    conflicts += 1
                
                # Check 2: exists k != i,j with D2_sel[j,k] == dij2
                row_j = D2_sel[j, :].copy()
                row_j[i] = -1
                row_j[j] = -1
                if np.any(row_j == dij2):
                    conflicts += 1
                
                # Check 3: exists k != i,j with D2_sel[i,k] == D2_sel[j,k]
                diff = D2_sel[i, :] - D2_sel[j, :]
                diff[i] = -1
                diff[j] = -1
                if np.any(diff == 0):
                    conflicts += 1
        
        return conflicts
    
    # Simulated annealing with vectorized compatibility checks
    best_selected = np.zeros(NUM_POINTS, dtype=bool)
    best_size = 0
    
    random.seed(42)
    np.random.seed(42)
    
    # Greedy initialization with vectorized checks
    selected = np.zeros(NUM_POINTS, dtype=bool)
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    
    for p in order:
        if time.time() > deadline - 30:
            break
        if is_compatible_vectorized(selected, p):
            selected[p] = True
    
    current_size = int(np.sum(selected))
    best_selected = selected.copy()
    best_size = current_size
    
    # Simulated annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute list of all points for random selection
    all_points = list(range(NUM_POINTS))
    
    sa_iteration = 0
    max_sa_iterations = 50000
    
    while time.time() < deadline - 15 and sa_iteration < max_sa_iterations:
        sa_iteration += 1
        
        # Decide move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random point
            p = random.randint(0, NUM_POINTS - 1)
            if not selected[p]:
                if is_compatible_vectorized(selected, p):
                    selected[p] = True
                    current_size += 1
                    if current_size > best_size:
                        best_selected = selected.copy()
                        best_size = current_size
                # Always accept (it's free)
        
        elif move_type < 0.6:
            # Remove 1, add 1
            if current_size > 0:
                selected_idx = np.where(selected)[0]
                if len(selected_idx) > 0:
                    remove_p = selected_idx[random.randint(0, len(selected_idx) - 1)]
                    old_selected = selected.copy()
                    selected[remove_p] = False
                    current_size -= 1
                    
                    # Try to add a random point
                    add_p = random.randint(0, NUM_POINTS - 1)
                    if not selected[add_p]:
                        if is_compatible_vectorized(selected, add_p):
                            selected[add_p] = True
                            current_size += 1
                            if current_size > best_size:
                                best_selected = selected.copy()
                                best_size = current_size
                        else:
                            # Revert
                            selected[:] = old_selected
                            current_size += 1
                    else:
                        # Revert
                        selected[:] = old_selected
                        current_size += 1
        
        elif move_type < 0.85:
            # Remove 1, add 2 (key move for escaping local minima)
            if current_size > 0:
                selected_idx = np.where(selected)[0]
                if len(selected_idx) > 0:
                    remove_p = selected_idx[random.randint(0, len(selected_idx) - 1)]
                    old_selected = selected.copy()
                    selected[remove_p] = False
                    current_size -= 1
                    
                    # Try to add two random points
                    added = 0
                    added_points = []
                    for _ in range(2):
                        if time.time() > deadline - 20:
                            break
                        add_p = random.randint(0, NUM_POINTS - 1)
                        if not selected[add_p] and add_p not in added_points:
                            if is_compatible_vectorized(selected, add_p):
                                selected[add_p] = True
                                added += 1
                                added_points.append(add_p)
                    
                    if added >= 1:
                        current_size += added
                        if current_size > best_size:
                            best_selected = selected.copy()
                            best_size = current_size
                    else:
                        # Revert
                        selected[:] = old_selected
                        current_size += 1
        
        else:
            # Random restart from a perturbed version of best solution
            if best_size > 0:
                selected = best_selected.copy()
                current_size = best_size
                # Remove a few random points
                num_remove = min(3, current_size)
                for _ in range(num_remove):
                    if current_size > 0:
                        selected_idx = np.where(selected)[0]
                        if len(selected_idx) > 0:
                            rp = selected_idx[random.randint(0, len(selected_idx) - 1)]
                            selected[rp] = False
                            current_size -= 1
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    # Final greedy pass on the best solution
    if time.time() < deadline - 10:
        selected = best_selected.copy()
        current_size = best_size
        for p in range(NUM_POINTS):
            if time.time() > deadline - 5:
                break
            if not selected[p]:
                if is_compatible_vectorized(selected, p):
                    selected[p] = True
                    current_size += 1
                    if current_size > best_size:
                        best_selected = selected.copy()
                        best_size = current_size
    
    # Return the best solution found
    result_idx = np.where(best_selected)[0]
    result = [(int(points[i, 0]), int(points[i, 1])) for i in result_idx]
    
    return result

# EVOLVE-BLOCK-END
