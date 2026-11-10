# EVOLVE-BLOCK-START
import numpy as np
import time

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position using a dynamic program
    over convex chains.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    total = 0

    for i in range(n - 6):
        later = pts[i + 1:]
        m = len(later)
        if m < 6:
            break

        dx = later[:, 0] - pts[i, 0]
        dy = later[:, 1] - pts[i, 1]
        angles = np.arctan2(dy, dx)
        angle_order = np.argsort(angles)
        spts = later[angle_order]
        m = len(spts)

        pi_pt = pts[i]
        dp = np.zeros((m, m), dtype=np.int64)
        
        for a in range(m - 1):
            for b in range(a + 1, m):
                cross = (spts[a, 0] - pi_pt[0]) * (spts[b, 1] - spts[a, 1]) - \
                        (spts[a, 1] - pi_pt[1]) * (spts[b, 0] - spts[a, 0])
                if cross > 1e-9:
                    dp[a, b] = 1
        
        for k in range(3, 7):
            new_dp = np.zeros((m, m), dtype=np.int64)
            for b in range(m - 1):
                for c in range(b + 1, m):
                    s = 0
                    for a in range(b):
                        if dp[a, b] > 0:
                            cross = (spts[b, 0] - spts[a, 0]) * (spts[c, 1] - spts[b, 1]) - \
                                    (spts[b, 1] - spts[a, 1]) * (spts[c, 0] - spts[b, 0])
                            if cross > 1e-9:
                                s += dp[a, b]
                    new_dp[b, c] = s
            dp = new_dp
        
        for a in range(m - 1):
            for b in range(a + 1, m):
                if dp[a, b] > 0:
                    cross = (spts[b, 0] - spts[a, 0]) * (pi_pt[1] - spts[b, 1]) - \
                            (spts[b, 1] - spts[a, 1]) * (pi_pt[0] - spts[b, 0])
                    if cross > 1e-9:
                        total += dp[a, b]
    
    return total

def _is_valid(points: np.ndarray) -> bool:
    """Check if the point set is valid."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(points >= -1e-6) or not np.all(points <= 1 + 1e-6):
        return False
    
    # Check no two points too close
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            if d2 < 1e-12:
                return False
    
    # Check no three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if abs(cross) < 1e-9:
                    return False
    
    return True

def _make_valid(points: np.ndarray) -> np.ndarray:
    """Ensure points are valid by adding small perturbations if needed."""
    points = points.copy()
    for _ in range(100):
        if _is_valid(points):
            return points
        # Add small random perturbations
        perturb = np.random.uniform(-0.001, 0.001, size=points.shape)
        points = points + perturb
        points = np.clip(points, 0.001, 0.999)
    # If still invalid, spread points more
    np.random.shuffle(points)
    points = np.clip(points, 0.001, 0.999)
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    best_points = None
    best_count = float('inf')
    
    # Generate grid positions for packing
    grid_size = 8
    grid_spacing = 1.0 / (grid_size - 1)
    grid_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            grid_positions.append((i * grid_spacing, j * grid_spacing))
    grid_positions = np.array(grid_positions)
    
    # Select 33 positions from the 64 available
    # We want to maximize collinearity, so we prefer rows/columns/diagonals
    # Let's try a few different subsets
    
    np.random.seed(42)
    
    # Strategy 1: Fill rows sequentially
    row_fill = []
    for i in range(grid_size):
        for j in range(grid_size):
            row_fill.append((i * grid_spacing, j * grid_spacing))
            if len(row_fill) == n:
                break
        if len(row_fill) == n:
            break
    row_fill = np.array(row_fill)
    
    # Strategy 2: Fill with more diagonal bias
    diag_fill = []
    for d in range(2 * grid_size - 1):
        for i in range(max(0, d - grid_size + 1), min(grid_size, d + 1)):
            j = d - i
            if 0 <= j < grid_size:
                diag_fill.append((i * grid_spacing, j * grid_spacing))
                if len(diag_fill) == n:
                    break
        if len(diag_fill) == n:
            break
    diag_fill = np.array(diag_fill)
    
    # Strategy 3: Random selection from grid
    random_grid = grid_positions[np.random.choice(len(grid_positions), n, replace=False)]
    
    # Strategy 4: Corner-heavy
    corner_heavy = []
    corners = [(0,0), (0, grid_spacing*(grid_size-1)), (grid_spacing*(grid_size-1), 0), (grid_spacing*(grid_size-1), grid_spacing*(grid_size-1))]
    for c in corners:
        corner_heavy.append(c)
    for pos in grid_positions:
        if len(corner_heavy) >= n:
            break
        if not any(np.allclose(pos, c) for c in corners):
            corner_heavy.append(tuple(pos))
    corner_heavy = np.array(corner_heavy[:n])
    
    configs = [row_fill, diag_fill, random_grid, corner_heavy]
    
    for cfg in configs:
        if time.time() > deadline:
            break
        
        pts = _make_valid(cfg)
        count = _count_convex7(pts)
        
        if count < best_count:
            best_count = count
            best_points = pts.copy()
        
        # Hill climbing with grid-aware moves
        step_size = 0.02
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline and no_improve < max_no_improve:
            idx = np.random.randint(n)
            
            # Mix of grid moves and continuous moves
            if np.random.random() < 0.3:
                # Move to a nearby grid position
                nearest_grid = grid_positions[np.argmin(np.sum((grid_positions - pts[idx])**2, axis=1))]
                # Perturb around nearest grid point
                new_pos = nearest_grid + np.random.normal(0, 0.01, size=2)
            else:
                # Continuous move
                new_pos = pts[idx] + np.random.normal(0, step_size, size=2)
            
            new_pts = pts.copy()
            new_pts[idx] = new_pos
            new_pts = np.clip(new_pts, 0.001, 0.999)
            
            if not _is_valid(new_pts):
                no_improve += 1
                continue
            
            new_count = _count_convex7(new_pts)
            
            if new_count < count:
                count = new_count
                pts = new_pts.copy()
                no_improve = 0
                step_size *= 0.97
            else:
                no_improve += 1
                if no_improve % 40 == 0:
                    step_size = min(step_size * 1.3, 0.05)
            
            # Update global best
            if count < best_count:
                best_count = count
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a valid configuration
        best_points = _make_valid(np.random.uniform(0.05, 0.95, size=(n, 2)))
    
    return best_points

# EVOLVE-BLOCK-END
