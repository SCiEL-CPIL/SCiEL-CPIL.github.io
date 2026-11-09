"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area calculation
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)

_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_triangle_area(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _is_inside(points):
    """Check if all points are inside or on the boundary of the triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, H)
    x = points[:, 0]
    y = points[:, 1]
    # Conditions for being inside:
    # y >= 0
    # y <= H * x  (below line from (0,0) to (0.5, H)) -> actually y <= 2*H*x for x in [0, 0.5]
    # Wait, the line from (0,0) to (0.5, H) has slope 2*H = sqrt(3). So y <= sqrt(3)*x.
    # The line from (1,0) to (0.5, H) has slope -sqrt(3). So y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x).
    # Also y >= 0.
    # So: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    cond1 = y >= -1e-9
    cond2 = y <= np.sqrt(3) * x + 1e-9
    cond3 = y <= np.sqrt(3) * (1 - x) + 1e-9
    return np.all(cond1 & cond2 & cond3)


def _objective(params):
    """
    Objective function for minimization: negative of min triangle area,
    plus a large penalty if points are outside the triangle.
    params: flat array of shape (2*N,)
    """
    points = params.reshape(N, 2)
    if not _is_inside(points):
        return 1e6  # Large penalty
    return -_min_triangle_area(points)


def _refine(points, max_time=30):
    """
    Refine a set of points using Nelder-Mead.
    Returns the best points found.
    """
    start_time = time.time()
    best_points = points.copy()
    best_val = _min_triangle_area(points)
    
    # Try a few different Nelder-Mead settings
    for _ in range(3):
        if time.time() - start_time > max_time:
            break
        res = minimize(_objective, points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.success and _is_inside(res.x.reshape(N, 2)):
            val = _min_triangle_area(res.x.reshape(N, 2))
            if val > best_val:
                best_val = val
                best_points = res.x.reshape(N, 2)
        # Add small perturbation to escape local minima
        if time.time() - start_time > max_time:
            break
        perturbed = best_points + np.random.normal(0, 0.01, (N, 2))
        # Project back inside triangle if needed (simple clipping)
        # For now, just try the perturbed version
        res2 = minimize(_objective, perturbed.flatten(), method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res2.success and _is_inside(res2.x.reshape(N, 2)):
            val2 = _min_triangle_area(res2.x.reshape(N, 2))
            if val2 > best_val:
                best_val = val2
                best_points = res2.x.reshape(N, 2)
    return best_points


def _generate_initial_points():
    """
    Generate a set of initial points using a grid-like structure with some randomness.
    """
    points = []
    # Use a few strategic starting configurations
    configs = [
        # Grid-like
        np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                  [0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3],
                  [0.3, 0.5], [0.5, 0.5]]),
        # Random
        None,
        # Corners and center
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H],
                  [0.25, H/4], [0.75, H/4], [0.5, H/2],
                  [0.125, H/8], [0.875, H/8], [0.375, 3*H/8],
                  [0.625, 3*H/8], [0.5, H/4]]),
    ]
    
    for config in configs:
        if config is None:
            # Generate random points
            pts = []
            for _ in range(N):
                u, v = np.random.random(2)
                if u + v < 1:
                    pts.append([u + 0.5*v, H*v])
            config = np.array(pts)
        if _is_inside(config):
            yield config
        else:
            # Try to fix by projecting
            fixed = config.copy()
            for i in range(N):
                x, y = fixed[i]
                if y < 0: y = 0
                if y > np.sqrt(3)*x: y = np.sqrt(3)*x * 0.99
                if y > np.sqrt(3)*(1-x): y = np.sqrt(3)*(1-x) * 0.99
                fixed[i] = [x, y]
            if _is_inside(fixed):
                yield fixed


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    best_points = None
    best_val = 0
    
    # Try multiple initial configurations
    np.random.seed(42)
    
    for config in _generate_initial_points():
        if time.time() > deadline - 5:
            break
        refined = _refine(config, max_time=min(20, deadline - time.time() - 5))
        val = _min_triangle_area(refined)
        if val > best_val:
            best_val = val
            best_points = refined
    
    # If we have time, do a few more random restarts
    while time.time() < deadline - 10:
        # Generate random points
        pts = []
        for _ in range(N):
            u, v = np.random.random(2)
            if u + v < 1:
                pts.append([u + 0.5*v, H*v])
        if len(pts) == N:
            config = np.array(pts)
            if _is_inside(config):
                refined = _refine(config, max_time=min(10, deadline - time.time() - 5))
                val = _min_triangle_area(refined)
                if val > best_val:
                    best_val = val
                    best_points = refined
    
    if best_points is None:
        # Fallback: return a simple grid
        best_points = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                [0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3],
                                [0.3, 0.5], [0.5, 0.5]])
    
    return best_points
