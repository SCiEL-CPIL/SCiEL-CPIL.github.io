"""Heilbronn triangle problem: place 11 points to maximize minimum triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_DEADLINE = 110.0  # safety margin before 120s kill
_START_TIME = None
_h = np.sqrt(3) / 2.0

# Triangle vertices in Cartesian coordinates
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, _h])

# Precompute triangle indices for speed
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _bary_to_cart(bary):
    """Convert barycentric coords (N, 3) to Cartesian (N, 2)."""
    # bary[i] = [lambda0, lambda1, lambda2] where sum = 1
    # cart = lambda0*V0 + lambda1*V1 + lambda2*V2
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2


def _cart_to_bary(pts):
    """Convert Cartesian (N, 2) to barycentric coords (N, 3)."""
    # Solve: p = a*V0 + b*V1 + c*V2, a+b+c=1
    # V0=(0,0), V1=(1,0), V2=(0.5,h)
    # x = b + 0.5*c, y = h*c => c = y/h, b = x - 0.5*c, a = 1-b-c
    c = pts[:, 1] / _h
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles from (N,2) array."""
    pts = points.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _soft_min_surrogate(bary_flat):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    Uses log-sum-exp trick for numerical stability.
    bary_flat: (N*3,) array of barycentric coordinates.
    """
    bary = bary_flat.reshape(N, 3)
    # Normalize to ensure they sum to 1 (safety)
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Soft-min via log-sum-exp: -soft_min = -(-beta * logsumexp(-areas/beta)) 
    # = beta * logsumexp(-areas/beta)
    # We want to MINIMIZE this, which maximizes the soft-min of areas.
    beta = 0.05
    # For stability: subtract max
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -beta * log_sum_exp  # This is the soft-min estimate
    return -soft_min_val  # Negate because we minimize


def _objective_nm(flat_pts):
    """Objective for Nelder-Mead: negative min area, with projection."""
    pts = flat_pts.reshape(N, 2)
    # Project to triangle
    for i in range(N):
        x, y = pts[i]
        y = max(0.0, min(_h, y))
        x_min = y / (2 * _h)
        x_max = 1 - y / (2 * _h)
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return -_min_triangle_area(pts)


def _make_initial_bary(seed, mode='random'):
    """Generate initial barycentric coordinate configurations."""
    rng = np.random.RandomState(seed)
    if mode == 'random':
        # Random points in triangle via barycentric
        bary = np.zeros((N, 3))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v > 1.0:
                u, v = rng.random(), rng.random()
            bary[i] = [1.0 - u - v, u, v]
        return bary
    elif mode == 'rows':
        # Structured: rows at different heights
        bary = np.zeros((N, 3))
        # Row 1 (bottom): y ~ 0.05*h => c ~ 0.05
        for j, x_frac in enumerate([0.15, 0.5, 0.85]):
            c = 0.05
            b = x_frac - 0.5 * c
            a = 1 - b - c
            bary[j] = [a, b, c]
        # Row 2: y ~ h/3 => c ~ 1/3
        for j, x_frac in enumerate([0.25, 0.5, 0.75]):
            c = 1.0 / 3.0
            b = x_frac - 0.5 * c
            a = 1 - b - c
            bary[3 + j] = [a, b, c]
        # Row 3: y ~ 2h/3 => c ~ 2/3
        for j, x_frac in enumerate([0.35, 0.5, 0.65]):
            c = 2.0 / 3.0
            b = x_frac - 0.5 * c
            a = 1 - b - c
            bary[6 + j] = [a, b, c]
        # Top: 2 points
        for j, x_frac in enumerate([0.4, 0.6]):
            c = 0.9
            b = x_frac - 0.5 * c
            a = 1 - b - c
            bary[9 + j] = [a, b, c]
        # Small perturbation
        bary += rng.uniform(-0.01, 0.01, size=(N, 3))
        bary = np.maximum(bary, 0.001)
        bary /= bary.sum(axis=1, keepdims=True)
        return bary
    elif mode == 'grid':
        bary = np.zeros((N, 3))
        positions = [
            (0.2, 0.1), (0.5, 0.1), (0.8, 0.1),
            (0.3, 0.35), (0.5, 0.35), (0.7, 0.35),
            (0.4, 0.6), (0.5, 0.6), (0.6, 0.6),
            (0.45, 0.8), (0.55, 0.8)
        ]
        for i, (x_frac, c_frac) in enumerate(positions):
            c = c_frac
            b = x_frac - 0.5 * c
            a = 1 - b - c
            bary[i] = [a, b, c]
        bary += rng.uniform(-0.01, 0.01, size=(N, 3))
        bary = np.maximum(bary, 0.001)
        bary /= bary.sum(axis=1, keepdims=True)
        return bary
    else:
        bary = np.zeros((N, 3))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v > 0.98:
                u, v = rng.random(), rng.random()
            bary[i] = [1.0 - u - v, u, v]
        return bary


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_points = None
    best_area = 0.0
    
    modes = ['rows', 'random', 'grid', 'random']
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for mode in modes:
        for seed in seeds:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE:
                break
            
            bary_init = _make_initial_bary(seed, mode)
            bary_flat = bary_init.ravel()
            
            # Evaluate initial
            pts_init = _bary_to_cart(bary_init)
            area = _min_triangle_area(pts_init)
            if area > best_area:
                best_area = area
                best_points = pts_init.copy()
            
            # Phase 1: L-BFGS-B on soft-min surrogate in barycentric space
            try:
                result = minimize(
                    _soft_min_surrogate,
                    bary_flat,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if result.fun < 0:  # Should be negative (we're minimizing -soft_min)
                    bary_opt = result.x.reshape(N, 3)
                    bary_opt = np.maximum(bary_opt, 0.0)
                    bary_opt /= bary_opt.sum(axis=1, keepdims=True)
                    opt_pts = _bary_to_cart(bary_opt)
                    opt_area = _min_triangle_area(opt_pts)
                    if opt_area > best_area:
                        best_area = opt_area
                        best_points = opt_pts.copy()
            except Exception:
                pass
            
            # Phase 2: Nelder-Mead polish on true objective in Cartesian space
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE:
                break
            
            if best_points is not None:
                try:
                    result = minimize(
                        _objective_nm,
                        best_points.ravel(),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.fun < 0:
                        opt_pts = result.x.reshape(N, 2)
                        # Final projection
                        for i in range(N):
                            x, y = opt_pts[i]
                            y = max(0.0, min(_h, y))
                            x_min = y / (2 * _h)
                            x_max = 1 - y / (2 * _h)
                            x = max(x_min, min(x_max, x))
                            opt_pts[i] = [x, y]
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
