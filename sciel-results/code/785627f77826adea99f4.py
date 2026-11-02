"""Robust Heilbronn-11 solver using Multi-start Nelder-Mead with strict interior projection."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Safety margin: Evaluator kills at 150s. We target 135s.
DEADLINE = 135.0

# Precompute triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _project_to_triangle(pts):
    """
    Project points strictly inside the triangle with a small margin.
    This ensures no point is on the boundary or outside.
    """
    pts = pts.copy()
    # Small margin to keep points strictly inside
    eps = 1e-5
    for i in range(len(pts)):
        x, y = pts[i]
        
        # Clip y to [eps, S3H - eps]
        y = np.clip(y, eps, S3H - eps)
        
        # For a given y, x must be in [y/S3 + eps, 1 - y/S3 - eps]
        x_min = y / S3 + eps
        x_max = 1.0 - y / S3 - eps
        
        # If the valid range is empty (due to eps being too large for very small y or large y),
        # fall back to centroid or clamp to center of available range
        if x_min >= x_max:
             # This shouldn't happen with eps=1e-5 and S3H~0.866, but safe guard
             x = 0.5
             y = S3H/2
        else:
            x = np.clip(x, x_min, x_max)
            
        pts[i] = [x, y]
    return pts

def _min_area(pts):
    """Calculate the minimum area of any triangle formed by the points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _objective(x_flat):
    """
    Objective function for Nelder-Mead.
    Minimizes the negative of the minimum area.
    Includes projection to ensure validity.
    """
    pts = x_flat.reshape(-1, 2)
    # Project to ensure points are inside
    pts = _project_to_triangle(pts)
    area = _min_area(pts)
    return -area

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = -1.0
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        # Final strict check
        pts = _project_to_triangle(pts)
        area = _min_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    def _time_left():
        return (DEADLINE - (time.time() - start_time)) > 5.0

    # 1. Generate Initial Configurations
    # Strategy: Symmetric grid + Random + Lattice subsets
    init_configs = []
    
    # Config 1: Symmetric "Star" pattern
    # Corners (slightly inset), Midpoints (slightly inset), Center
    p0 = np.array([0.05, 0.05]) # Inset corner
    p1 = np.array([0.95, 0.05])
    p2 = np.array([0.5, S3H - 0.05])
    pm0 = np.array([0.5, 0.05])
    pm1 = np.array([0.75, S3H/2 - 0.02])
    pm2 = np.array([0.25, S3H/2 - 0.02])
    pc = np.array([0.5, S3H/3])
    
    # Add 4 more points to reach 11
    # Place them symmetrically around the center
    r = 0.15
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    # Adjust center for equilateral triangle
    cx, cy = 0.5, S3H/3
    extra_pts = []
    for a in angles:
        # Approximate offset
        dx = r * np.cos(a)
        dy = r * np.sin(a)
        extra_pts.append([cx + dx, cy + dy])
        
    init1 = np.vstack([p0, p1, p2, pm0, pm1, pm2, pc] + extra_pts[:4])
    init_configs.append(init1)
    
    # Config 2: Random
    for _ in range(5):
        pts = rng.random((N, 2))
        # Map to triangle
        # Use barycentric coords for random sampling
        u = rng.random(N)
        v = rng.random(N)
        w = 1 - u - v
        mask = w < 0
        while np.any(mask):
             u[mask] = rng.random(np.sum(mask))
             v[mask] = rng.random(np.sum(mask))
             w = 1 - u - v
             mask = w < 0
        x = v * 1.0 + w * 0.5
        y = w * S3H
        init2 = np.column_stack((x, y))
        init_configs.append(init2)
        
    # Config 3: Lattice-based (Grid)
    # Create a grid of points and pick 11
    grid_size = 6
    grid_pts = []
    for i in range(grid_size):
        for j in range(grid_size):
             # Map grid to triangle
             # Use barycentric grid
             u = i / (grid_size - 1)
             v = j / (grid_size - 1)
             w = 1 - u - v
             if w >= -1e-9:
                 x = v * 1.0 + w * 0.5
                 y = w * S3H
                 grid_pts.append([x, y])
                 
    # Shuffle and take 11
    rng.shuffle(grid_pts)
    if len(grid_pts) >= N:
        init3 = np.array(grid_pts[:N])
        init_configs.append(init3)

    # 2. Optimize each configuration
    for i, init in enumerate(init_configs):
        if not _time_left():
            break
            
        # Project initial points
        init = _project_to_triangle(init)
        _update_best(init)
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _objective,
                init.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.success:
                opt_pts = res.x.reshape(-1, 2)
                opt_pts = _project_to_triangle(opt_pts)
                _update_best(opt_pts)
                
        except Exception:
            pass
            
    # 3. Local Refinement of the Best Point
    if best_pts is not None and _time_left():
        # Run a few more Nelder-Mead iterations from the best point with tighter tolerances
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.success:
                final_pts = res.x.reshape(-1, 2)
                final_pts = _project_to_triangle(final_pts)
                _update_best(final_pts)
        except Exception:
            pass

    # 4. Final Validation and Return
    if best_pts is None:
        # Fallback: Centroid and corners
        best_pts = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.5, 0.7],
            [0.5, 0.1], [0.75, 0.35], [0.25, 0.35],
            [0.5, 0.3], [0.3, 0.2], [0.7, 0.2],
            [0.4, 0.4], [0.6, 0.4]
        ])
        
    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
