import numpy as np
import time
from scipy.optimize import minimize

def _compute_2d_area(X, Y, c):
    """
    Compute the area of the union of 64 squares at height c.
    X: array of 8 x-base positions.
    Y: array of 8 y-base positions.
    c: scalar height.
    """
    N = 8
    side = 1.0 / N
    half_side = side / 2.0
    
    # Centers
    cx = X + c * np.arange(N) / N
    cy = Y + c * np.arange(N) / N
    
    # Left/Right, Bottom/Top
    xl = cx - half_side
    xr = cx + half_side
    yb = cy - half_side
    yt = cy + half_side
    
    # Create list of squares: (x1, x2, y1, y2)
    # We need the union area.
    # Collect all unique x and y coordinates.
    all_x = np.unique(np.concatenate([xl, xr]))
    all_y = np.unique(np.concatenate([yb, yt]))
    
    # If unique coords are too many, this is slow. But N=8, so 16 unique x's?
    # No, xl and xr are distinct for different i generally.
    # So ~16 unique x's, ~16 unique y's.
    # Grid size: 15 x 15 = 225 cells.
    # This is very fast!
    
    # Create meshgrid of cell centers? No, we need to check if cell is inside any square.
    # Actually, we can just iterate through the cells.
    # Cell (i, j) is [all_x[i], all_x[i+1]] x [all_y[j], all_y[j+1]]
    
    area = 0.0
    # Precompute square ranges in index space?
    # Or just loop. 225 cells * 64 squares = 14400 checks. Very fast.
    
    for i in range(len(all_x) - 1):
        x1, x2 = all_x[i], all_x[i+1]
        w = x2 - x1
        if w < 1e-12: continue
        for j in range(len(all_y) - 1):
            y1, y2 = all_y[j], all_y[j+1]
            h = y2 - y1
            if h < 1e-12: continue
            
            # Check if this cell is covered by ANY square
            # A cell is covered if it intersects any square.
            # Since the grid is formed by the square edges, a cell is either fully inside or fully outside a specific square?
            # Yes, because the cell boundaries are a subset of the square boundaries.
            # So we just need to check if the cell's *interior* is inside a square.
            # Or simply, check if the cell's bottom-left corner is inside?
            # No, if the cell is [x1, x2] and square is [sx1, sx2], and x1, x2 are from the set of square edges,
            # then either the cell is completely inside, completely outside, or on the boundary.
            # We can check if the cell's center is inside any square.
            
            cx_cell = 0.5 * (x1 + x2)
            cy_cell = 0.5 * (y1 + y2)
            
            # Vectorized check: is (cx_cell, cy_cell) inside any square?
            # Square k is (cx[k], cy[k]) +/- half_side
            # Condition: |cx_cell - cx[k]| <= half_side AND |cy_cell - cy[k]| <= half_side
            # We can do this for all k at once?
            # dist_x = np.abs(cx_cell - cx)
            # dist_y = np.abs(cy_cell - cy)
            # covered = np.any((dist_x <= half_side) & (dist_y <= half_side))
            
            # This is 64 operations per cell. 225 * 64 = 14400 ops.
            # In Python loop, this might be slow if called 1000 times.
            # Let's vectorize over cells.
            
            # Actually, let's just do the full vectorization for the whole grid at once.
            pass
            
    # Let's rewrite _compute_2d_area to be fully vectorized.
    # 1. Get all_x, all_y.
    # 2. Create meshgrid of cell centers.
    # 3. For each square, mark cells it covers.
    
    return area

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    side = 1.0 / N
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        X = params[:N]
        Y = params[N:]
        
        # Integrate area over c
        # Use Gauss-Legendre or simple trapezoidal with many points
        # 50 points should be enough for smooth-ish area function?
        # The area function is piecewise smooth. Breakpoints occur when square edges align.
        # With 64 squares, there are many breakpoints.
        # Let's use 100 points.
        
        c_vals = np.linspace(0, 1, 100)
        total_area = 0.0
        
        for c in c_vals:
            # Calculate area at c
            # Vectorized grid method
            cx = X + c * np.arange(N) / N
            cy = Y + c * np.arange(N) / N
            
            xl = cx - side/2
            xr = cx + side/2
            yb = cy - side/2
            yt = cy + side/2
            
            all_x = np.unique(np.concatenate([xl, xr]))
            all_y = np.unique(np.concatenate([yb, yt]))
            
            # Cell centers
            # nx = len(all_x) - 1
            # ny = len(all_y) - 1
            # cx_grid, cy_grid = np.meshgrid(0.5*(all_x[:-1]+all_x[1:]), 0.5*(all_y[:-1]+all_y[1:]), indexing='ij')
            
            # Instead of meshgrid, we can check coverage for each cell.
            # But let's try to vectorize the "is covered" check.
            
            # For each cell (i, j), we need to know if it is covered by any square.
            # Cell (i, j) is defined by x-range [all_x[i], all_x[i+1]] and y-range [all_y[j], all_y[j+1]].
            # It is covered if it intersects any square.
            # Since grid lines are square edges, a cell is covered if its *interior* is covered.
            # Which is equivalent to checking if the cell's bottom-left corner (or any point) is inside a square.
            # Actually, if a cell is not covered, it means NO square contains it.
            
            # Let's create a boolean mask of size (nx, ny)
            # Initialize to False
            # For each square k, find the range of cells it covers.
            # Square k covers x in [xl[k], xr[k]] and y in [yb[k], yt[k]].
            # Find indices i such that all_x[i] >= xl[k] and all_x[i+1] <= xr[k].
            # This is a searchsorted operation.
            
            nx = len(all_x) - 1
            ny = len(all_y) - 1
            covered = np.zeros((nx, ny), dtype=bool)
            
            for k in range(N):
                # x range
                i_start = np.searchsorted(all_x, xl[k], side='left')
                i_end = np.searchsorted(all_x, xr[k], side='right') - 1
                
                # y range
                j_start = np.searchsorted(all_y, yb[k], side='left')
                j_end = np.searchsorted(all_y, yt[k], side='right') - 1
                
                # Clamp
                i_start = max(0, i_start)
                i_end = min(nx - 1, i_end)
                j_start = max(0, j_start)
                j_end = min(ny - 1, j_end)
                
                if i_start <= i_end and j_start <= j_end:
                    covered[i_start:i_end+1, j_start:j_end+1] = True
            
            # Calculate area
            # Widths of x cells
            wx = np.diff(all_x)
            wy = np.diff(all_y)
            # Area of each cell is wx[i] * wy[j]
            # Total area = sum(covered * wx[i] * wy[j])
            # We can do: area = np.sum(covered * np.outer(wx, wy))
            
            area = np.sum(covered * np.outer(wx, wy))
            total_area += area
            
        # Trapezoidal integration
        # We have 100 points.
        # Integral = sum * (1/100) approx?
        # Better: np.trapz(areas, c_vals)
        # But we accumulated in loop. Let's store areas.
        
        # This loop is slow. Let's vectorize the c loop?
        # No, the grid structure changes with c.
        
        # Let's just use the accumulated sum with simple rectangle rule for speed,
        # or store in array.
        # For now, return total_area / len(c_vals) * 1.0 (rough average)
        # Actually, let's do it properly.
        
        return total_area / len(c_vals) # Approximation

    # This is too slow. 100 c-values * (sort + loop) per evaluation.
    # Sort is O(N log N). Loop is O(N).
    # 100 * (8 log 8 + 8) ~ 1000 ops.
    # But the inner loop `for k in range(N)` is 8 iterations.
    # searchsorted is fast.
    # `np.sum(covered * np.outer(wx, wy))` is fast.
    # The bottleneck is the Python loop over c.
    
    # Let's try to optimize the objective function.
    
    # Initial guess
    X0 = np.linspace(0, 1, N) - 0.5
    Y0 = np.linspace(0, 1, N) - 0.5
    params0 = np.concatenate([X0, Y0])
    
    # Optimize
    res = minimize(objective, params0, method='Powell', options={'maxiter': 500})
    
    # Return best
    if res.fun < 1e10:
        X_opt = res.x[:N]
        Y_opt = res.x[N:]
    else:
        X_opt = X0
        Y_opt = Y0
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
