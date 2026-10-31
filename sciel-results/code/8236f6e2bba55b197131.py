"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def _check_time():
    return time.time() - _START_TIME < _TIME_LIMIT

def _compute_radius(circles):
    """Compute the maximum uniform radius scaling factor for circles.
    circles: (N, 3) array [x, y, r]
    Returns the largest t such that all circles of radius t*r[i] fit in a
    rectangle of perimeter 4 and are disjoint.
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # We want to find max scale s such that:
    # 1. All circles fit in rectangle [0, W] x [0, H] with 2(W+H) <= 4, i.e. W+H <= 2
    # 2. Circles are disjoint: |ci - cj| >= s*(ri + rj)
    
    # For a given s, the required width is max(s*r + x) - min(s*r + x)
    # Actually, the bounding box of the scaled circles:
    # x_min = min(x_i - s*r_i), x_max = max(x_i + s*r_i)
    # Width = x_max - x_min
    # Similarly for height.
    # Perimeter constraint: 2*(width + height) <= 4 => width + height <= 2
    
    # Disjointness: for all i < j, sqrt((xi-xj)^2 + (yi-yj)^2) >= s*(ri + rj)
    
    def _feasible(s):
        # Check perimeter
        xmin = np.min(x - s * r)
        xmax = np.max(x + s * r)
        ymin = np.min(y - s * r)
        ymax = np.max(y + s * r)
        width = xmax - xmin
        height = ymax - ymin
        if width + height > 2.0 + 1e-12:
            return False
        # Check disjointness
        if n <= 1:
            return True
        dists = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
        min_dists = np.min(dists, axis=1)  # This isn't quite right, need pairwise
        # Actually, for each pair, dist >= s*(ri+rj)
        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] < s * (r[i] + r[j]) - 1e-12:
                    return False
        return True
    
    # Bisection for max s
    lo, hi = 0.0, 2.0
    # First check if hi is feasible
    if not _feasible(hi):
        # Find upper bound
        while _feasible(hi):
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            if _feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo
    else:
        return hi

def _objective(circles_flat):
    """Objective to maximize: sum of radii after optimal scaling.
    We'll instead minimize negative of a smooth surrogate.
    """
    n = len(circles_flat) // 3
    circles = circles_flat.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Compute the max scale s
    # For smoothness, use soft-min for constraints
    # Perimeter: width + height <= 2
    xmin = np.min(x - r)  # at s=1
    xmax = np.max(x + r)
    ymin = np.min(y - r)
    ymax = np.max(y + r)
    width = xmax - xmin
    height = ymax - ymin
    perim_violation = np.maximum(0.0, width + height - 2.0)
    
    # Disjointness: dist_ij >= r_i + r_j
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            req = r[i] + r[j]
            if d < req:
                penalty += (req - d)**2
    
    return penalty + 100 * perim_violation**2

def _initial_grid(n_circles):
    """Create initial circle positions in a grid pattern."""
    # Use a grid layout
    cols = int(np.ceil(np.sqrt(n_circles * 2)))  # wider than tall
    rows = int(np.ceil(n_circles / cols))
    
    # Rectangle aspect ratio: let's start with W=1.5, H=0.5 (W+H=2)
    W, H = 1.5, 0.5
    
    circles = []
    for i in range(n_circles):
        row = i // cols
        col = i % cols
        # Adjust for last row
        actual_cols_in_row = min(cols, n_circles - row * cols)
        x = (col + 0.5) / actual_cols_in_row * W
        y = (row + 0.5) / rows * H
        r = 0.5 / cols * 0.9
        circles.append([x, y, r])
    
    return np.array(circles)

def _optimize_positions(circles, max_iter=50):
    """Optimize circle positions to maximize the minimum constraint slack."""
    n = len(circles)
    
    def _score(cflat):
        """Higher is better: we want to maximize the minimum scaled radius."""
        c = cflat.reshape(n, 3)
        x, y, r = c[:, 0], c[:, 1], c[:, 2]
        
        # Max scale s
        xmin = np.min(x - r)
        xmax = np.max(x + r)
        ymin = np.min(y - r)
        ymax = np.max(y + r)
        width = xmax - xmin
        height = ymax - ymin
        s_perim = 2.0 / (width + height) if (width + height) > 0 else 1.0
        
        s_pair = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                s_pair = min(s_pair, d / (r[i] + r[j]))
        
        s = min(s_perim, s_pair)
        return s
    
    # Start from current positions
    x0 = circles[:, :2].flatten()
    
    def _neg_score(xflat):
        c = circles.copy()
        c[:, :2] = xflat.reshape(n, 2)
        return -_score(c.flatten())
    
    # Use Nelder-Mead for optimization
    best_x = x0
    best_score = _score(circles.flatten())
    
    for _ in range(max_iter):
        if not _check_time():
            break
        res = minimize(_neg_score, best_x, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.x is not None:
            new_score = -res.fun
            if new_score > best_score:
                best_score = new_score
                best_x = res.x
    
    circles[:, :2] = best_x.reshape(n, 2)
    return circles

def _expand_radii(circles, iterations=30):
    """Iteratively expand radii to fill available space."""
    n = len(circles)
    
    for _ in range(iterations):
        if not _check_time():
            break
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # For each circle, find how much it can grow
        # Limited by:
        # 1. Other circles
        # 2. Rectangle boundaries (after centering)
        
        # First, center the configuration
        xmin = np.min(x - r)
        xmax = np.max(x + r)
        ymin = np.min(y - r)
        ymax = np.max(y + r)
        width = xmax - xmin
        height = ymax - ymin
        scale = 2.0 / (width + height)
        
        # Scale everything
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        x = (x - cx) * scale
        y = (y - cy) * scale
        r = r * scale
        
        # Now try to grow each circle
        for i in range(n):
            # Max growth limited by neighbors
            max_grow = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                # r_i + grow_i + r_j <= d => grow_i <= d - r_i - r_j
                grow_limit = d - r[i] - r[j]
                if grow_limit < max_grow:
                    max_grow = grow_limit
            # Also limited by perimeter (approximate)
            # Just apply a fraction of the max growth
            grow = 0.3 * max(0, max_grow)
            r[i] += grow
        
        # Re-center and normalize
        xmin = np.min(x - r)
        xmax = np.max(x + r)
        ymin = np.min(y - r)
        ymax = np.max(y + r)
        width = xmax - xmin
        height = ymax - ymin
        scale = 2.0 / (width + height)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        x = (x - cx) * scale
        y = (y - cy) * scale
        r = r * scale
        
        circles[:, 0] = x
        circles[:, 1] = y
        circles[:, 2] = r
    
    return circles

def _final_polish(circles, tol=1e-10):
    """High-precision bisection polish."""
    n = len(circles)
    
    def _feasible(s):
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        xmin = np.min(x - s * r)
        xmax = np.max(x + s * r)
        ymin = np.min(y - s * r)
        ymax = np.max(y + s * r)
        if (xmax - xmin) + (ymax - ymin) > 2.0 + tol:
            return False
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < s * (r[i] + r[j]) - tol:
                    return False
        return True
    
    lo, hi = 0.0, 1.0
    # Ensure hi is infeasible or find it
    while _feasible(hi):
        hi *= 1.5
    for _ in range(50):
        if not _check_time():
            break
        mid = (lo + hi) / 2
        if _feasible(mid):
            lo = mid
        else:
            hi = mid
    
    s = lo
    circles[:, 2] *= s
    return circles

def construct():
    nonlocal _START_TIME
    _START_TIME = time.time()
    
    n = N
    # Start with grid initialization
    circles = _initial_grid(n)
    
    # Optimize positions
    if _check_time():
        circles = _optimize_positions(circles, max_iter=20)
    
    # Expand radii
    if _check_time():
        circles = _expand_radii(circles, iterations=40)
    
    # Final polish
    if _check_time():
        circles = _final_polish(circles, tol=1e-10)
    
    # Ensure valid output: perimeter <= 4
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    xmin = np.min(x - r)
    xmax = np.max(x + r)
    ymin = np.min(y - r)
    ymax = np.max(y + r)
    width = xmax - xmin
    height = ymax - ymin
    perim = 2 * (width + height)
    if perim > 4.0:
        scale = 4.0 / perim
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        circles[:, 0] = (circles[:, 0] - cx) * scale
        circles[:, 1] = (circles[:, 1] - cy) * scale
        circles[:, 2] *= scale
    
    return circles
# EVOLVE-BLOCK-END
