import numpy as np
import time
from itertools import product

def compute_union_area_at_c(c, x_positions, y_positions, N=8):
    """
    Compute the area of the union of rectangles at height c.
    x_positions: array of shape (N,) for x-base positions (one per i)
    y_positions: array of shape (N,) for y-base positions (one per j)
    """
    # Generate x-intervals
    # i in 0..N-1
    i = np.arange(N)
    x_left = x_positions + c * i / N
    x_right = x_left + 1.0 / N
    # Generate y-intervals
    j = np.arange(N)
    y_left = y_positions + c * j / N
    y_right = y_left + 1.0 / N
    
    # We have N x N rectangles.
    # To compute the union area, we can use a sweep line or grid.
    # Since N is small, we can just collect all x-coordinates and y-coordinates
    # and sum the areas of the elementary cells that are covered.
    # But the rectangles are axis-aligned.
    # A simpler way: 
    # The union of rectangles in 2D can be computed by:
    # 1. Collect all unique x-coordinates from all left and right edges.
    # 2. Collect all unique y-coordinates from all bottom and top edges.
    # 3. For each cell in the grid formed by these coordinates, check if it is covered by at least one rectangle.
    # 4. Sum the areas of covered cells.
    
    x_edges = np.unique(np.concatenate([x_left, x_right]))
    y_edges = np.unique(np.concatenate([y_left, y_right]))
    
    # Number of cells is at most (2N-1) x (2N-1) = 15x15 = 225 cells.
    # For each cell, we need to check if it is inside any of the 64 rectangles.
    # This is 225 * 64 = 14400 checks per c.
    # With 100 c-values, that's 1.4M checks. Very fast in Python if vectorized.
    
    # Vectorized check:
    # Create a grid of cell centers or just use the edges.
    # Cell (px, py) is [x_edges[px], x_edges[px+1]] x [y_edges[py], y_edges[py+1]]
    # It is covered if there exists (i,j) such that:
    # x_left[i] <= x_edges[px] AND x_right[i] >= x_edges[px+1]
    # y_left[j] <= y_edges[py] AND y_right[j] >= y_edges[py+1]
    
    # Let's precompute the coverage for each cell.
    # x_edges has length Lx, y_edges has length Ly.
    # Cells are (Lx-1) x (Ly-1).
    
    # For each cell, we can check against all i and j.
    # But we can do it smarter.
    # For a given x-interval [x_edges[px], x_edges[px+1]], which i's cover it?
    # This is a 1D interval coverage problem.
    
    # Let's just use a simple loop for clarity, as N=8 is small.
    area = 0.0
    for px in range(len(x_edges) - 1):
        x0 = x_edges[px]
        x1 = x_edges[px+1]
        if x1 <= x0: continue
        # Find i's that cover [x0, x1]
        # x_left[i] <= x0 and x_right[i] >= x1
        cover_i = np.where((x_left <= x0) & (x_right >= x1))[0]
        if len(cover_i) == 0:
            continue
        for py in range(len(y_edges) - 1):
            y0 = y_edges[py]
            y1 = y_edges[py+1]
            if y1 <= y0: continue
            # Find j's that cover [y0, y1]
            cover_j = np.where((y_left <= y0) & (y_right >= y1))[0]
            if len(cover_j) == 0:
                continue
            # If there is at least one i and one j, the cell is covered by the rectangle (i,j)
            # Wait, the rectangle is (i,j). So we need the same i and j?
            # No, the union is over all (i,j). So if there exists ANY i and ANY j such that the cell is in rect(i,j), it is covered.
            # But rect(i,j) is the product of interval i and interval j.
            # So the cell is covered if (exists i in cover_i) AND (exists j in cover_j).
            # Yes, because if i covers x-part and j covers y-part, then (i,j) covers the cell.
            area += (x1 - x0) * (y1 - y0)
            
    return area

def compute_volume(x_positions, y_positions, N=8, num_c=50):
    """
    Estimate the volume by integrating the area over c in [0,1].
    """
    c_vals = np.linspace(0, 1, num_c, endpoint=False) + 0.5/num_c
    areas = [compute_union_area_at_c(c, x_positions, y_positions, N) for c in c_vals]
    return np.mean(areas) * 1.0 # height is 1

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_positions = keich.copy()
    y_positions = keich.copy()
    
    # Local search
    best_vol = compute_volume(x_positions, y_positions)
    
    # Time limit guard
    start_time = time.time()
    max_time = 600.0 # 600s limit, leave some margin
    
    step_size = 0.01
    for iteration in range(1000):
        if time.time() - start_time > max_time - 30:
            break
            
        # Randomly pick a position to perturb
        idx = np.random.randint(0, N)
        is_x = np.random.rand() > 0.5
        
        if is_x:
            old_val = x_positions[idx]
            x_positions[idx] += np.random.uniform(-step_size, step_size)
        else:
            old_val = y_positions[idx]
            y_positions[idx] += np.random.uniform(-step_size, step_size)
            
        new_vol = compute_volume(x_positions, y_positions)
        
        if new_vol < best_vol:
            best_vol = new_vol
        else:
            # Revert
            if is_x:
                x_positions[idx] = old_val
            else:
                y_positions[idx] = old_val
                
        # Reduce step size occasionally
        if iteration % 100 == 0 and iteration > 0:
            step_size *= 0.5

    # Construct final positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_positions[i_flat]
    base_y = y_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
