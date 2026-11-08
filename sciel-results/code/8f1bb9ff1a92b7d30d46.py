"""Heilbronn triangle problem: place 11 points to maximize minimum triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_DEADLINE = 110.0  # safety margin before 120s kill
_START_TIME = None

# Precompute triangle indices for speed
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])
_h = np.sqrt(3) / 2.0


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles from flat array."""
    pts = points.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _project_to_triangle(pts):
    """Project 2D points into the equilateral triangle with vertices (0,0), (1,0), (0.5, h)."""
    h = _h
    for i in range(len(pts)):
        x, y = pts[i]
        # Clip y to [0, h]
        y = max(0.0, min(h, y))
        # For a given y, x must satisfy: x >= y/h (left edge) and x <= 1 - y/h (right edge)
        # Left edge: from (0,0) to (0.5, h): x = y/(2h) * ... wait
        # Left edge: (0,0) to (0.5, h): parametric x = 0.5*t, y = h*t => x = y/(2h)
        # Right edge: (1,0) to (0.5, h): parametric x = 1 - 0.5*t, y = h*t => x = 1 - y/(2h)
        x_min = y / (2 * h)
        x_max = 1 - y / (2 * h)
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _objective_flat(flat_pts):
    """Objective to minimize: negative of min triangle area."""
    pts = _project_to_triangle(flat_pts.reshape(N, 2).copy())
    return -_min_triangle_area(pts.ravel())


def _make_initial_points(seed, mode='random'):
    """Generate initial point configurations."""
    rng = np.random.RandomState(seed)
    if mode == 'random':
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0 and u > 0.005 and v > 0.005:
                x = u + 0.5 * v
                y = _h * v
                pts.append([x, y])
        return np.array(pts)
    elif mode == 'rows':
        # Try row-based patterns
        pts = []
        # Row at y=0: 3 points
        for x in [0.1, 0.5, 0.9]:
            pts.append([x, 0.01])
        # Row at y=h/3: 3 points
        for x in [0.2, 0.5, 0.8]:
            pts.append([x, _h / 3])
        # Row at y=2h/3: 3 points
        for x in [0.3, 0.5, 0.7]:
            pts.append([x, 2 * _h / 3])
        # Top: 2 points
        pts.append([0.4, 0.95 * _h])
        pts.append([0.6, 0.95 * _h])
        # Add small random perturbation
        pts = np.array(pts) + rng.uniform(-0.02, 0.02, size=(N, 2))
        return _project_to_triangle(pts)
    elif mode == 'grid':
        pts = []
        # 3x3 grid plus 2 extra
        for i in range(3):
            for j in range(3):
                x = 0.1 + 0.4 * i + 0.1 * j
                y = 0.1 + 0.3 * j
                pts.append([x, y])
        pts.append([0.5, 0.9])
        pts.append([0.3, 0.7])
        pts = np.array(pts[:N])
        return _project_to_triangle(pts)
    else:
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.99:
                x = u + 0.5 * v
                y = _h * v
                pts.append([x, y])
        return np.array(pts)


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_points = None
    best_area = 0.0
    
    # Try multiple initial configurations and optimization methods
    modes = ['rows', 'random', 'grid', 'random']
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for mode in modes:
        for seed in seeds:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE:
                break
            
            pts = _make_initial_points(seed, mode)
            flat = pts.ravel()
            
            # Evaluate initial
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts.copy()
            
            # Try Nelder-Mead optimization
            try:
                result = minimize(
                    _objective_flat,
                    flat,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.success or result.fun < 0:
                    opt_pts = _project_to_triangle(result.x.reshape(N, 2).copy())
                    opt_area = _min_triangle_area(opt_pts)
                    if opt_area > best_area:
                        best_area = opt_area
                        best_points = opt_pts.copy()
            except Exception:
                pass
            
            # Try Powell optimization (more robust for non-smooth)
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE:
                break
            
            try:
                result = minimize(
                    _objective_flat,
                    flat,
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-8}
                )
                if result.success or result.fun < 0:
                    opt_pts = _project_to_triangle(result.x.reshape(N, 2).copy())
                    opt_area = _min_triangle_area(opt_pts)
                    if opt_area > best_area:
                        best_area = opt_area
                        best_points = opt_pts.copy()
            except Exception:
                pass
        
        # Check time between modes
        if time.time() - _START_TIME > _DEADLINE:
            break
    
    # If no points found, fall back to random
    if best_points is None:
        rng = np.random.RandomState(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, _h * v])
        best_points = np.array(pts)
    
    return best_points
