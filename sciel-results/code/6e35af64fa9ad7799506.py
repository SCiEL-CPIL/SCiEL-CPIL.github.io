"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _project_to_triangle(pts):
    """Project points onto the equilateral triangle (0,0), (1,0), (0.5, H).
    Uses barycentric coordinates: for triangle with vertices A=(0,0), B=(1,0), C=(0.5,H),
    a point P = (x,y) has barycentric coords:
    w0 = 1 - 2*y/H  (weight for A... wait let me recalculate)
    
    Actually for triangle A=(0,0), B=(1,0), C=(0.5, H):
    The barycentric coordinates (a, b, c) with a+b+c=1 give P = a*A + b*B + c*C
    So x = b + 0.5*c, y = H*c
    Thus c = y/H, b = x - 0.5*c = x - 0.5*y/H, a = 1 - b - c = 1 - x - 0.5*y/H
    
    Wait: x = 0*a + 1*b + 0.5*c = b + 0.5c
    y = 0*a + 0*b + H*c = H*c
    So c = y/H
    b = x - 0.5c = x - 0.5*y/H
    a = 1 - b - c = 1 - (x - 0.5*y/H) - y/H = 1 - x - 0.5*y/H
    
    For P to be inside: a>=0, b>=0, c>=0
    i.e., 1-x-0.5*y/H >= 0, x-0.5*y/H >= 0, y/H >= 0
    i.e., x + 0.5*y/H <= 1, x >= 0.5*y/H, y >= 0
    """
    pts = np.array(pts, dtype=float, copy=True)
    x, y = pts[:, 0], pts[:, 1]
    
    # Barycentric coords
    c = y / _H
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    # Clip to non-negative
    a = np.maximum(a, 0)
    b = np.maximum(b, 0)
    c = np.maximum(c, 0)
    
    # Renormalize (they should sum to 1 already if point was inside, but after clipping may not)
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    # Convert back to Cartesian
    new_x = b + 0.5 * c
    new_y = _H * c
    
    return np.column_stack([new_x, new_y])


def _objective(x_flat):
    """Negative min triangle area (for minimization)."""
    pts = x_flat.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Generate a symmetric initial configuration for 11 points."""
    # Use a hexagonal-like arrangement with symmetry
    # Center point, then 6 points in a ring, then 4 more
    pts = []
    # Center
    pts.append([0.5, _H / 3.0])
    
    # Inner ring (6 points at radius r1, angles 0, 60, 120, 180, 240, 300)
    r1 = 0.15
    for i in range(6):
        angle = i * np.pi / 3.0
        px = 0.5 + r1 * np.cos(angle)
        py = _H / 3.0 + r1 * np.sin(angle)
        pts.append([px, py])
    
    # Outer ring (4 points)
    r2 = 0.28
    for i in range(4):
        angle = i * np.pi / 2.0 + np.pi / 4.0
        px = 0.5 + r2 * np.cos(angle)
        py = _H / 3.0 + r2 * np.sin(angle)
        pts.append([px, py])
    
    return np.array(pts)


def _random_initial_config(rng):
    """Generate a random initial configuration inside the triangle."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    return np.array(pts)


def construct():
    """Place 11 points on or inside the unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Symmetric config
    initial_configs.append(_symmetric_initial_config())
    
    # Several random configs
    for _ in range(8):
        initial_configs.append(_random_initial_config(rng))
    
    # Also try some structured configs
    # Grid-like
    grid_pts = []
    for i in range(3):
        for j in range(3):
            # Place on a triangular grid
            x = 0.2 + 0.3 * i
            y = 0.1 + 0.25 * j
            # Check if inside
            if y >= 0 and x >= 0.5 * y / _H and x + 0.5 * y / _H <= 1.0:
                grid_pts.append([x, y])
    if len(grid_pts) >= N:
        initial_configs.append(np.array(grid_pts[:N]))
    
    best_pts = None
    best_score = -np.inf
    
    # Define bounds for SLSQP (loose, we project anyway)
    bounds = [(0, 1)] * (2 * N)
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            pts = result.x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            score = _min_triangle_area(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # If we have time, do a second pass with more iterations on the best config
    if best_pts is not None and time.time() < deadline - 20.0:
        try:
            x0 = best_pts.flatten()
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            pts = result.x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Final verification and return
    if best_pts is None:
        # Fallback: return the symmetric config
        best_pts = _symmetric_initial_config()
    
    # Ensure all points are inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts
