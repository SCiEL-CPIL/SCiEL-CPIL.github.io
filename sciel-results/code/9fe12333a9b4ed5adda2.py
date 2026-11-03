"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize

N = 200


def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _hex_grid_params_to_points(params):
    """
    Convert optimization parameters to circle centers.
    params:
        [0]: s_x (horizontal spacing)
        [1]: s_y (vertical spacing)
        [2]: x_offset
        [3]: y_offset
        [4:]: row_shifts (per row x-shift, length = n_rows)
    """
    s_x = params[0]
    s_y = params[1]
    x_off = params[2]
    y_off = params[3]
    row_shifts = params[4:]
    
    n_rows = len(row_shifts)
    points = []
    
    # Determine number of points per row
    # We need to distribute N points into n_rows rows.
    # Let base_cols = N // n_rows, rem = N % n_rows
    # First 'rem' rows have base_cols + 1 points, others have base_cols.
    base_cols = N // n_rows
    rem = N % n_rows
    
    y_coords = []
    cols_per_row = []
    
    for r in range(n_rows):
        y = -0.5 + y_off + r * s_y
        y_coords.append(y)
        if r < rem:
            cols_per_row.append(base_cols + 1)
        else:
            cols_per_row.append(base_cols)
            
    # Adjust x-centering for each row based on its specific number of columns
    for r in range(n_rows):
        n_cols = cols_per_row[r]
        shift = row_shifts[r]
        # Center the row of n_cols points around x=0 (plus global offset)
        # Spacing is s_x.
        # x_i = - (n_cols - 1) * s_x / 2 + i * s_x
        # Total width = (n_cols - 1) * s_x
        # To fit in [-0.5, 0.5], we need to handle boundaries carefully in the objective,
        # but for generation, we just place them relative to center.
        start_x = -((n_cols - 1) * s_x) / 2.0
        for c in range(n_cols):
            x = start_x + c * s_x + x_off + shift
            points.append([x, y_coords[r]])
            
    return np.array(points)


def _neg_radius_hex(params):
    """Objective function: negative of the max radius for a hex grid defined by params."""
    try:
        points = _hex_grid_params_to_points(params)
    except Exception:
        return 1000.0
        
    if not np.all(np.isfinite(points)):
        return 1000.0
        
    # Check bounds
    if np.any(points < -0.5) or np.any(points > 0.5):
        # Penalty for being out of bounds
        # We want to maximize radius, so if out of bounds, radius is effectively 0 or negative.
        # Return a large positive value (minimizing neg_radius) to push points back in.
        # Actually, better to just compute the "radius" which will be negative if out of bounds?
        # The score is min(min_pair/2, min_boundary). If min_boundary < 0, radius is negative.
        # We want to minimize -radius. So if radius is negative, -radius is positive.
        # We want to minimize this. So we should return -radius.
        pass
        
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
        
    radius = min(min_pair / 2.0, min_boundary)
    return -radius


def _optimize_hex_grid(n_rows, time_limit=10.0):
    """
    Optimize a hexagonal grid with n_rows.
    """
    start_time = time.time()
    
    # Initial guess
    # Approximate area per circle: 1.0 / N
    # Hex packing density is ~0.9069.
    # Area of hex cell ~ s_x * s_y * sqrt(3)/2 ? No, standard hex grid.
    # Let's estimate s_x and s_y.
    # If we have n_rows, vertical span is (n_rows-1)*s_y + diameter.
    # Horizontal span is approx n_cols * s_x + diameter.
    # n_cols ~ N / n_rows.
    
    n_cols_avg = N / n_rows
    
    # Fit in square:
    # (n_cols_avg - 1) * s_x + 2*r <= 1  => s_x <= (1 - 2r) / (n_cols_avg - 1)
    # (n_rows - 1) * s_y + 2*r <= 1  => s_y <= (1 - 2r) / (n_rows - 1)
    # Also hex constraint: s_y = s_x * sqrt(3) / 2
    
    # Let's just start with a reasonable grid.
    s_x_init = 1.0 / n_cols_avg
    s_y_init = s_x_init * np.sqrt(3) / 2.0
    
    # If s_y_init * (n_rows - 1) > 1, we need to shrink.
    if n_rows > 1:
        max_s_y = 1.0 / (n_rows - 1)
        if s_y_init > max_s_y:
            s_y_init = max_s_y
            s_x_init = s_y_init * 2.0 / np.sqrt(3)
            
    x_off_init = 0.0
    y_off_init = 0.0
    
    # Row shifts: alternate 0 and s_x/2 for hex packing
    row_shifts_init = []
    for r in range(n_rows):
        if r % 2 == 0:
            row_shifts_init.append(0.0)
        else:
            row_shifts_init.append(s_x_init / 2.0)
            
    x0 = np.array([s_x_init, s_y_init, x_off_init, y_off_init] + row_shifts_init)
    
    # Bounds
    # s_x, s_y > 0. Upper bound 1.0.
    # x_off, y_off in [-0.5, 0.5]
    # row_shifts in [-0.5, 0.5]
    
    bounds = []
    bounds.append((1e-4, 1.0)) # s_x
    bounds.append((1e-4, 1.0)) # s_y
    bounds.append((-0.5, 0.5)) # x_off
    bounds.append((-0.5, 0.5)) # y_off
    for _ in range(n_rows):
        bounds.append((-0.5, 0.5)) # row_shifts
        
    # Optimization
    best_res = None
    best_fun = np.inf
    
    # Try a few random perturbations to initial state
    for attempt in range(3):
        if time.time() - start_time > time_limit:
            break
            
        if attempt == 0:
            x0_try = x0
        else:
            # Perturb
            noise = np.random.uniform(-0.05, 0.05, size=x0.shape)
            x0_try = x0 + noise
            # Ensure positive spacing
            x0_try[0] = max(1e-4, x0_try[0])
            x0_try[1] = max(1e-4, x0_try[1])
            
        try:
            res = minimize(_neg_radius_hex, x0_try, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
        except Exception:
            continue
            
    if best_res is not None:
        points = _hex_grid_params_to_points(best_res.x)
        r = _compute_radius(points)
        return points, r
    else:
        return None, 0.0


def construct():
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Optimized Hex Grids with varying row counts
    # Try row counts that might fit well.
    # sqrt(200) ~ 14.14. So 13, 14, 15 rows are good candidates.
    # Also try 12, 16.
    row_counts = [12, 13, 14, 15, 16]
    
    for n_rows in row_counts:
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time() - 5
        time_per_grid = min(10.0, remaining / max(1, len(row_counts)))
        
        points, r = _optimize_hex_grid(n_rows, time_limit=time_per_grid)
        
        if points is not None and r > best_r:
            best_r = r
            best_points = points.copy()
            
    # Fallback if optimization failed or returned bad result
    if best_points is None or best_r < 0.03:
        # Simple square grid fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        sq_points = grid[:N]
        r = _compute_radius(sq_points)
        if r > best_r:
            best_r = r
            best_points = sq_points.copy()
            
    if best_points is None:
        best_points = np.zeros((N, 2))
        return best_points
        
    return best_points
