"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(n, w, h, cols, rows):
    """Create a grid of n circles in rectangle w x h."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            r = 0.5 / max(cols, rows) * min(w, h) - 1e-3
            circles.append((cx, cy, r))
    # If we have more than n, trim
    circles = circles[:n]
    # If we have fewer, add more at center with small radius
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.01))
    return circles

def _local_optimize(circles, w, h, deadline):
    """
    Local optimization using Nelder-Mead to maximize sum of radii.
    Variables: (cx_i, cy_i, r_i) for each circle i.
    """
    n = len(circles)
    x0 = []
    for cx, cy, r in circles:
        x0.extend([cx, cy, r])
    x0 = np.array(x0)
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        circles = []
        for i in range(n):
            cx = x[i * 3]
            cy = x[i * 3 + 1]
            r = x[i * 3 + 2]
            circles.append((cx, cy, r))
        
        # Penalty-based objective: return negative sum of radii if invalid
        for cx, cy, r in circles:
            if r <= 0:
                return 0.0
            if cx - r < -1e-9 or cx + r > w + 1e-9 or cy - r < -1e-9 or cy + r > h + 1e-9:
                return 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0.0
        
        return _total_radius(circles)
    
    try:
        res = minimize(
            lambda x: -objective(x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if objective(res.x) > 0:
            best_circles = []
            for i in range(n):
                best_circles.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
            if _validate(best_circles, w, h):
                return best_circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # Try various aspect ratios
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios to try: (w, h) where w+h=2
    # w ranges from 0.5 to 1.5 (symmetric, so w<=h means w<=1)
    aspect_ratios = [
        (1.0, 1.0),    # square
        (0.8, 1.2),
        (0.7, 1.3),
        (0.6, 1.4),
        (0.9, 1.1),
        (0.75, 1.25),
    ]
    
    # Grid configurations: (cols, rows)
    grid_configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (4, 5),
        (5, 4),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline:
            break
        
        for cols, rows in grid_configs:
            if time.time() > deadline:
                break
            
            # Create initial grid
            circles = _grid_init(N, w, h, cols, rows)
            
            # Validate initial grid
            if not _validate(circles, w, h):
                continue
            
            score = _total_radius(circles)
            if score > best_score:
                best_score = score
                best_circles = circles
            
            # Try local optimization
            if time.time() < deadline - 5.0:  # Need some time left
                optimized = _local_optimize(circles, w, h, deadline)
                if optimized is not None:
                    opt_score = _total_radius(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = optimized
    
    # If we found a good result, return it
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: simple grid in 1x1 square
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
