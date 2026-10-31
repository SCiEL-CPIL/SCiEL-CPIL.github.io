"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _evaluate(arr):
    """Compute perimeter of bounding box and sum of radii.
    Returns (perimeter, sum_radii). Lower perimeter is better for feasibility,
    higher sum_radii is the objective."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Check disjointness (interiors disjoint means dist >= r_i + r_j)
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return (np.inf, 0.0)
    
    w = x.max() - x.min()
    h = y.max() - y.min()
    perim = 2.0 * (w + h)
    
    # Objective: maximize sum of radii, but we need perim <= 4
    # We'll normalize: scale so that perim = 4, then sum of scaled radii is the score
    # Actually, the task says "bounding rectangle has perimeter at most 4"
    # and we maximize sum of radii. So we want to find circles with perim <= 4
    # and maximize sum(r).
    
    # The score is just sum(r) given perim <= 4.
    # For evaluation during search, we can use a penalty or scaling approach.
    # Let's return the raw values.
    return (perim, np.sum(r))


def _scale_to_perim4(arr):
    """Scale the configuration so that its bounding box has perimeter exactly 4.
    Returns the scaled array."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    w = x.max() - x.min()
    h = y.max() - y.min()
    perim = 2.0 * (w + h)
    
    if perim <= 0:
        return arr.copy()
    
    scale = 4.0 / perim
    return np.column_stack([x * scale, y * scale, r * scale])


def _sum_radii_perim4(arr):
    """Compute sum of radii after scaling to perimeter 4."""
    scaled = _scale_to_perim4(arr)
    return np.sum(scaled[:, 2])


def construct():
    t_start = time.time()
    
    best_score = 0.0
    best_arr = None
    
    # Initial configuration: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    # Use a grid of size x_size x y_size that fits N circles
    # Try different aspect ratios
    
    def make_grid(nx, ny, w, h):
        """Create nx x ny grid of circles in rectangle [0,w] x [0,h]."""
        xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx) if nx > 1 else np.array([w / 2])
        ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny) if ny > 1 else np.array([h / 2])
        
        circles = []
        for i in range(nx):
            for j in range(ny):
                if len(circles) >= N:
                    break
                circles.append([xs[i], ys[j], 0.0])  # radius to be determined
        return np.array(circles[:N])
    
    def compute_radii(arr):
        """Given centers, compute max radius such that circles are disjoint and within bounds.
        Actually, let's just set radius as half the minimum distance to other circles and boundaries."""
        x = arr[:, 0]
        y = arr[:, 1]
        w = x.max() - x.min()
        h = y.max() - y.min()
        
        # For each circle, max radius is half the min distance to other circles
        # and to the bounding box boundaries
        radii = np.zeros(N)
        for i in range(N):
            max_r = np.inf
            # Distance to boundaries
            min_x = x.min()
            max_x = x.max()
            min_y = y.min()
            max_y = y.max()
            max_r = min(max_r, x[i] - min_x, max_x - x[i], y[i] - min_y, max_y - y[i])
            
            # Distance to other circles
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                max_r = min(max_r, d / 2.0)
            
            radii[i] = max_r
        return radii
    
    # Strategy: Use a grid layout, then optimize positions to maximize the minimum radius
    # which corresponds to maximizing the sum of radii after scaling to perim 4.
    
    # Let's try a direct optimization approach:
    # Place circles in a grid, then use local search to improve the minimum gap.
    
    best_score = 0.0
    best_arr = None
    
    def evaluate_score(arr):
        """Score = sum of radii after scaling to perim 4, if valid."""
        x = arr[:, 0]
        y = arr[:, 1]
        
        w = x.max() - x.min()
        h = y.max() - y.min()
        if w <= 0 or h <= 0:
            return 0.0
        
        perim = 2.0 * (w + h)
        if perim <= 0:
            return 0.0
        
        # Compute radii as half the min distance to neighbors and boundaries
        radii = np.zeros(N)
        for i in range(N):
            max_r = np.inf
            min_x = x.min()
            max_x = x.max()
            min_y = y.min()
            max_y = y.max()
            max_r = min(max_r, x[i] - min_x, max_x - x[i], y[i] - min_y, max_y - y[i])
            
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                max_r = min(max_r, d / 2.0)
            
            radii[i] = max_r
        
        # Check if circles with these radii are actually disjoint (they should be by construction)
        # Scale to perim 4
        scale = 4.0 / perim
        scaled_radii = radii * scale
        return np.sum(scaled_radii)
    
    # Generate initial grid configurations
    def generate_grid(nx, ny):
        """Generate a grid configuration with nx columns and ny rows."""
        # Aspect ratio: make cells roughly square
        # Total perimeter is 4, so w + h = 2
        # For nx x ny grid, cell size is w/nx by h/ny
        # To make cells square: w/nx = h/ny => w = h * nx/ny
        # w + h = 2 => h * nx/ny + h = 2 => h = 2*ny/(nx+ny), w = 2*nx/(nx+ny)
        
        w = 2.0 * nx / (nx + ny)
        h = 2.0 * ny / (nx + ny)
        
        xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
        ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
        
        circles = []
        for j in range(ny):
            for i in range(nx):
                if len(circles) >= N:
                    break
                circles.append([xs[i], ys[j]])
        return np.array(circles[:N])
    
    # Try different grid layouts
    layouts = []
    for nx in range(1, N + 1):
        for ny in range(1, N + 1):
            if nx * ny >= N:
                layouts.append((nx, ny))
    
    # Sort by aspect ratio closeness to 1 (square cells are usually better)
    def cell_aspect(nx, ny):
        # w = 2nx/(nx+ny), h = 2ny/(nx+ny)
        # cell_w = w/nx = 2/(nx+ny), cell_h = h/ny = 2/(nx+ny)
        # So cells are always square! Aspect ratio is 1.
        return 0.0
    
    # All grids have square cells, so let's just try a few reasonable ones
    # 5x5=25 >= 21, 4x6=24 >= 21, 3x7=21, 6x4=24, 7x3=21
    candidate_layouts = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (2, 11), (11, 2), (1, 21), (21, 1)]
    
    best_score = 0.0
    best_arr = None
    
    for nx, ny in candidate_layouts:
        arr = generate_grid(nx, ny)
        score = evaluate_score(arr)
        
        # Local optimization: perturb positions to improve score
        # Use a simple hill-climbing / gradient-free approach
        arr_opt = arr.copy()
        current_score = score
        
        # Try to improve by moving circles
        np.random.seed(42)
        max_iters = 5000
        step_size = 0.01
        
        for it in range(max_iters):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            
            # Pick a random circle and perturb it
            i = np.random.randint(N)
            dx = np.random.randn() * step_size
            dy = np.random.randn() * step_size
            
            arr_new = arr_opt.copy()
            arr_new[i, 0] += dx
            arr_new[i, 1] += dy
            
            new_score = evaluate_score(arr_new)
            
            if new_score > current_score:
                arr_opt = arr_new
                current_score = new_score
                step_size *= 1.01  # increase step slightly
            else:
                step_size *= 0.95  # decrease step
                step_size = max(step_size, 1e-6)
        
        if current_score > best_score:
            best_score = current_score
            best_arr = arr_opt
        
        if time.time() - t_start > TIME_LIMIT - 5:
            break
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
        best_arr = np.array(cells[:N])
    
    # Final: compute radii and scale to perimeter 4
    x = best_arr[:, 0]
    y = best_arr[:, 1]
    
    w = x.max() - x.min()
    h = y.max() - y.min()
    perim = 2.0 * (w + h)
    
    if perim <= 0:
        # Degenerate case
        return np.zeros((N, 3))
    
    scale = 4.0 / perim
    
    # Compute radii
    radii = np.zeros(N)
    for i in range(N):
        max_r = np.inf
        min_x = x.min()
        max_x = x.max()
        min_y = y.min()
        max_y = y.max()
        max_r = min(max_r, x[i] - min_x, max_x - x[i], y[i] - min_y, max_y - y[i])
        
        for j in range(N):
            if i == j:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            max_r = min(max_r, d / 2.0)
        
        radii[i] = max_r
    
    # Scale everything
    x_scaled = x * scale
    y_scaled = y * scale
    r_scaled = radii * scale
    
    # Small epsilon to ensure strict disjointness
    r_scaled = r_scaled * (1 - 1e-9)
    
    result = np.column_stack([x_scaled, y_scaled, r_scaled])
    
    # Verify: check that bounding box perimeter <= 4
    w_final = x_scaled.max() - x_scaled.min()
    h_final = y_scaled.max() - y_scaled.min()
    perim_final = 2.0 * (w_final + h_final)
    
    # If perimeter is slightly over 4 due to numerical issues, scale down a bit more
    if perim_final > 4.0:
        extra_scale = 4.0 / perim_final
        result = result * extra_scale
        # Recompute radii conservatively
        for i in range(N):
            max_r = np.inf
            min_x = result[:, 0].min()
            max_x = result[:, 0].max()
            min_y = result[:, 1].min()
            max_y = result[:, 1].max()
            max_r = min(max_r, result[i, 0] - min_x, max_x - result[i, 0], 
                       result[i, 1] - min_y, max_y - result[i, 1])
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((result[i, 0] - result[j, 0])**2 + (result[i, 1] - result[j, 1])**2)
                max_r = min(max_r, d / 2.0)
            result[i, 2] = min(result[i, 2], max_r * (1 - 1e-9))
    
    return result
