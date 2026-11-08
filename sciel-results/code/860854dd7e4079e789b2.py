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
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _soft_min_area(pts, temperature=0.001):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and helps with gradient-based optimization."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft minimum: -log-sum-exp(-areas/T) * T
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse * temperature


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Constraints:
      y >= 0
      y <= sqrt(3) * x        (left edge from origin to top)
      y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)  (right edge)
    """
    pts = pts.copy()
    # Clip y >= 0
    pts[pts[:, 1] < 0, 1] = 0.0
    # Clip to left edge: y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    pts[mask_left, 0] = min_x_left[mask_left]
    # Clip to right edge: y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    pts[mask_right, 0] = max_x_right[mask_right]
    # Final clip for numerical safety
    pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1)
    return pts


def _objective_smooth(x, temperature=0.001):
    """Negative soft-min triangle area (smooth, to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_soft_min_area(pts, temperature)


def _objective_exact(x):
    """Negative exact min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points."""
    cx, cy = 0.5, _H / 3
    r = 0.25
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [[cx, cy]]
    for a in angles:
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        pts.append([px, py])
    pts.append([0.15, 0.1])
    pts.append([0.85, 0.1])
    pts.append([0.5, 0.7])
    pts.append([0.35, 0.35])
    arr = np.array(pts[:N], dtype=float)
    return _project_to_triangle(arr).flatten()


def _random_initial_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        while u + v > 1.0:
            u, v = rng.random(), rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def _nelder_mead_optimize(x0, max_time=15.0, temperature=0.001):
    """Run Nelder-Mead optimization with time-based stopping.
    Returns (best_x, best_val_exact)."""
    start = time.time()
    
    # Use smooth objective for Nelder-Mead
    def obj(x):
        return _objective_smooth(x, temperature)
    
    best_x = x0.copy()
    best_val_smooth = obj(best_x)
    best_val_exact = _objective_exact(best_x)
    
    def callback(xk):
        nonlocal best_x, best_val_smooth, best_val_exact
        val_s = obj(xk)
        if val_s < best_val_smooth:
            best_x = xk.copy()
            best_val_smooth = val_s
            best_val_exact = _objective_exact(xk)
        return time.time() - start > max_time
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'callback': callback}
        )
        if res.fun < best_val_smooth:
            best_x = res.x
            best_val_smooth = res.fun
            best_val_exact = _objective_exact(best_x)
    except Exception:
        pass
    
    return best_x, best_val_exact


def _slsqp_polish(x0, max_iter=300):
    """Final SLSQP polish with exact objective."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _objective_exact(best_x)
    
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_x = None
    best_val = np.inf
    
    # Start 1: symmetric configuration
    x0 = _symmetric_initial_config()
    x, val = _nelder_mead_optimize(x0, max_time=10.0, temperature=0.001)
    best_x, best_val = x, val
    
    # Additional random starts with Nelder-Mead on smooth objective
    n_starts = 30
    for seed in range(n_starts):
        if time.time() > deadline - 25:
            break
        x0 = _random_initial_config(seed + 100)
        x, val = _nelder_mead_optimize(x0, max_time=8.0, temperature=0.001)
        if val < best_val:
            best_x, best_val = x, val
    
    # Final refinement: polish with SLSQP on exact objective
    if time.time() < deadline - 10:
        best_x, best_val = _slsqp_polish(best_x, max_iter=300)
    
    # One more polish if time permits
    if time.time() < deadline - 5:
        best_x, best_val = _slsqp_polish(best_x, max_iter=200)
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
