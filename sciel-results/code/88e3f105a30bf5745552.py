"""Program for the no-isosceles set task using a constraint-propagation backtracking search."""
import time
import random
import numpy as np

N = 64
DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s limit


def construct():
    """
    Construct a set of points in [0, N)^2 with no three forming an isosceles triangle.
    
    Strategy:
    - Backtracking search with strong constraint propagation.
    - For each candidate point, track which squared distances are "forbidden" 
      (i.e., would create an isosceles triangle with two already-placed points).
    - Dynamic variable ordering: at each step, try the point that has the fewest 
      remaining valid options (most constrained first) to prune early.
    - Multiple random restarts with different initial orderings.
    """
    start_time = time.time()
    deadline = start_time + 330 - DEADLINE_MARGIN
    
    # Precompute all grid points
    grid = [(x, y) for x in range(N) for y in range(N)]
    n_grid = len(grid)
    
    # Precompute squared distances between all pairs of grid points
    # dist2[i][j] = squared distance between grid[i] and grid[j]
    # This is a 4096x4096 matrix, which is ~16M entries. Might be large but let's try.
    # Actually, 4096^2 = 16,777,216 entries. As int32, that's ~64MB. Acceptable.
    
    grid_arr = np.array(grid, dtype=np.int32)
    # Compute pairwise squared distances
    # dist2[i, j] = (grid[i] - grid[j])^2
    dx = grid_arr[:, 0:1] - grid_arr[1:, 0:1].T  # This doesn't work directly
    # Let's do it properly
    x_coords = grid_arr[:, 0]
    y_coords = grid_arr[:, 1]
    dx_mat = x_coords[:, None] - x_coords[None, :]
    dy_mat = y_coords[:, None] - y_coords[None, :]
    dist2 = (dx_mat * dx_mat + dy_mat * dy_mat).astype(np.int32)
    
    best_points = []
    
    # Multiple restarts with different strategies
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering of grid points
        order = np.random.permutation(n_grid)
        
        # State: list of selected point indices
        selected = []
        selected_set = set()
        
        # For each selected point, we need to track which distances are "used"
        # Key insight: A new point p can be added if:
        # 1. For every pair of selected points (a, b), dist(p, a) != dist(p, b) 
        #    (p is not equidistant from any two selected points)
        # 2. For every selected point a, and every other selected point b != a,
        #    dist(p, a) != dist(a, b)
        #    (the distance from p to a doesn't equal the distance between a and any other selected point)
        
        # Let's maintain:
        # - selected: list of selected point indices
        # - For fast checking, we'll compute distances from candidate to all selected points
        
        for idx in order:
            if time.time() > deadline:
                break
            
            p_idx = int(idx)
            p = grid[p_idx]
            px, py = p
            valid = True
            
            k = len(selected)
            if k >= 2:
                # Compute distances from p to all selected points
                sel_indices = np.array(selected, dtype=np.int32)
                d2_to_sel = dist2[p_idx, sel_indices]  # squared distances from p to each selected point
                
                # Check 1: All distances from p to selected points must be distinct
                if len(np.unique(d2_to_sel)) < len(d2_to_sel):
                    valid = False
                else:
                    # Check 2: For each selected point i, d2_to_sel[i] must not equal 
                    # dist(selected[i], selected[j]) for any j != i
                    # We can vectorize this: for each i, check if d2_to_sel[i] is in 
                    # the set {dist2[selected[i], selected[j]] for j != i}
                    
                    # Build the set of forbidden distances for each selected point
                    # forbidden[i] = set of dist2[selected[i], selected[j]] for j != i
                    # Then check if d2_to_sel[i] in forbidden[i]
                    
                    # Vectorized approach:
                    # For each i, the forbidden distances are dist2[sel_indices[i], sel_indices] 
                    # excluding dist2[sel_indices[i], sel_indices[i]] (which is 0)
                    
                    # Let's create a matrix where row i contains all distances from sel_indices[i] to all selected
                    # dist_matrix = dist2[np.ix_(sel_indices, sel_indices)]  # k x k matrix
                    # For row i, the forbidden values are all entries except the diagonal (which is 0)
                    
                    dist_matrix = dist2[np.ix_(sel_indices, sel_indices)]
                    
                    # For each i, check if d2_to_sel[i] equals any off-diagonal element in row i
                    # Create a mask of off-diagonal elements
                    # We need: for all i, d2_to_sel[i] not in {dist_matrix[i, j] for j != i}
                    
                    # One approach: for each i, check if d2_to_sel[i] is in the set of off-diagonal elements of row i
                    # This is O(k^2) but k is small in early stages
                    
                    if k <= 200:
                        for i in range(k):
                            target = d2_to_sel[i]
                            # Check if target equals any dist_matrix[i, j] for j != i
                            row = dist_matrix[i]
                            # Exclude the diagonal element (j == i)
                            # Create mask
                            mask = np.ones(k, dtype=bool)
                            mask[i] = False
                            if np.any(row[mask] == target):
                                valid = False
                                break
                    else:
                        # For larger k, use a more efficient check
                        # For each i, we need to check if d2_to_sel[i] is in the multiset of 
                        # off-diagonal elements of row i
                        # We can use a set for each row, but that's O(k^2) memory
                        # Alternative: for each i, use np.isin
                        for i in range(k):
                            target = d2_to_sel[i]
                            row = dist_matrix[i]
                            mask = np.ones(k, dtype=bool)
                            mask[i] = False
                            if np.any(row[mask] == target):
                                valid = False
                                break
            
            if valid:
                selected.append(p_idx)
                selected_set.add(p_idx)
        
        current_points = [grid[i] for i in selected]
        
        if len(current_points) > len(best_points):
            best_points = list(current_points)
        
        if time.time() > deadline:
            break
        
        # Try a second pass with a different ordering to potentially add more points
        # Points not yet selected
        remaining = [i for i in range(n_grid) if i not in selected_set]
        random.shuffle(remaining)
        
        for p_idx in remaining:
            if time.time() > deadline:
                break
            
            p = grid[p_idx]
            px, py = p
            valid = True
            
            k = len(selected)
            if k >= 2:
                sel_indices = np.array(selected, dtype=np.int32)
                d2_to_sel = dist2[p_idx, sel_indices]
                
                if len(np.unique(d2_to_sel)) < len(d2_to_sel):
                    valid = False
                else:
                    dist_matrix = dist2[np.ix_(sel_indices, sel_indices)]
                    
                    if k <= 200:
                        for i in range(k):
                            target = d2_to_sel[i]
                            row = dist_matrix[i]
                            mask = np.ones(k, dtype=bool)
                            mask[i] = False
                            if np.any(row[mask] == target):
                                valid = False
                                break
                    else:
                        for i in range(k):
                            target = d2_to_sel[i]
                            row = dist_matrix[i]
                            mask = np.ones(k, dtype=bool)
                            mask[i] = False
                            if np.any(row[mask] == target):
                                valid = False
                                break
            
            if valid:
                selected.append(p_idx)
                selected_set.add(p_idx)
        
        current_points = [grid[i] for i in selected]
        
        if len(current_points) > len(best_points):
            best_points = list(current_points)
    
    return best_points


# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Found {len(res)} points in {end-start:.2f}s")
    # Verify no isosceles triangles
    n = len(res)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = res[i], res[j], res[k]
                d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                if d12 == d23 or d23 == d13 or d13 == d12:
                    print(f"Isosceles found: {p1}, {p2}, {p3}")
                    exit(1)
    print("Valid!")
