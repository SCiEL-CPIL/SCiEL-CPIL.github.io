import numpy as np
import time
import sys

def _get_grid():
    # Grid points: (-3 + 0.01i, -1.5 + 0.01j)
    # i = 0..449, j = 0..299
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _check_in_base(points, x, y, a_deg):
    # points: (N, 2) array of [x, y]
    # Returns boolean mask of points inside the base corridor placed at (x,y) rotated by a_deg
    
    a_rad = np.deg2rad(a_deg)
    # Rotation matrix R(-a)
    # We need R(-a) * (w - (x,y)) to be in base
    # Base: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    
    # Rotate by -a
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)
    
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    
    # Check base conditions
    # Rect 1: 0 <= u <= 1, -3 <= v <= 1
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Rect 2: -3 <= u <= 1, 0 <= v <= 1
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def _evaluate_path(path, X, Y):
    # path: (N, 3)
    # X, Y: flattened grid coords
    points = np.column_stack((X, Y))
    
    # Construct full path
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full_path = np.vstack([start, path, end])
    
    # Densify
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i+1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            densified.append(interp)
            
    # Evaluate intersection
    # Start with all True
    sofa_mask = np.ones(points.shape[0], dtype=bool)
    
    for p in densified:
        x, y, a = p
        mask = _check_in_base(points, x, y, a)
        sofa_mask &= mask
        # Early exit if empty
        if not np.any(sofa_mask):
            return 0.0
            
    return np.sum(sofa_mask) * 0.0001
