"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

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
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = np.array(pts, dtype=float)
    for _ in range(20):
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        pts[:, 0] = np.maximum(pts[:, 0], 0)
        
        mask2 = pts[:, 1] > _SQRT3 * pts[:, 0]
        if np.any(mask2):
            pts[mask2, 1] = _SQRT3 * pts[mask2, 0]
            
        mask3 = pts[:, 1] > _SQRT3 - _SQRT3 * pts[:, 0]
        if np.any(mask3):
            pts[mask3, 1] = _SQRT3 - _SQRT3 * pts[mask3, 0]
            
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        
    return pts


def _soft_min(values, beta=100.0):
    """Smooth approximation of the minimum."""
    min_val = np.min(values)
    shifted = values - min_val
    # Avoid overflow
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
    sum_exp = np.sum(np.exp(-beta * shifted))
    return min_val - np.log(sum_exp) / beta


def _penalty(points):
    """Penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - _SQRT3 * x)
    viol3 = np.maximum(0, y - (_SQRT3 - _SQRT3 * x))
    return np.sum(d1**2) + np.sum(viol2**2) + np.sum(viol3**2)


def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: negative soft-min of triangle areas + penalty.
    This is differentiable and suitable for L-BFGS-B.
    """
    pts = x.reshape(N, 2)
    # Don't project during optimization; use penalty instead for smoothness
    # But keep points roughly in bounds via penalty
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min of areas
    sm = _soft_min(areas, beta)
    
    # Penalty for being outside triangle
    pen = _penalty(pts)
    
    return -(sm) + 10.0 * pen


def _true_objective(x):
    """Negative min triangle area (to be minimized)."""
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


def _generate_structured_start(seed):
    """Generate structured initial configurations using barycentric coordinates."""
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(0.5, 2.0, size=3)
    bary = rng.dirichlet(alpha, size=N)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = _H * bary[:, 2]
    pts = np.column_stack([x, y])
    return _project_to_triangle(pts).flatten()


def _lbfgsb_optimize(x0, max_time=8.0):
    """
    Run L-BFGS-B on the smooth surrogate objective.
    Returns (best_x, best_val_true).
    """
    start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    # Multi-stage beta for soft-min
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > start + max_time - 1.0:
            break
        
        obj = lambda x: _smooth_objective(x, beta)
        
        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < obj(best_x):
                cand = res.x
                cand_val = _true_objective(cand)
                if cand_val < best_val:
                    best_x = cand
                    best_val = cand_val
        except Exception:
            pass
    
    return best_x, best_val


def _slsqp_polish(x0, max_time=5.0):
    """
    Final polish with SLSQP on the true (non-smooth) objective.
    Returns (best_x, best_val).
    """
    start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    try:
        res = minimize(
            _true_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, best_val


def _simulated_annealing(x0, max_time=3.0, T0=0.008, cooling=0.97, rng=None):
    """Simulated annealing to escape local optima."""
    if rng is None:
        rng = np.random.RandomState()
    
    start = time.time()
    current_x = x0.copy()
    current_val = _true_objective(current_x)
    best_x = current_x.copy()
    best_val = current_val
    T = T0
    
    while time.time() < start + max_time and T > 1e-10:
        idx = rng.randint(N)
        delta = rng.normal(0, T, size=2)
        new_x = current_x.copy()
        new_x[2*idx] += delta[0]
        new_x[2*idx+1] += delta[1]
        
        new_pts = _project_to_triangle(new_x.reshape(N, 2))
        new_val = _true_objective(new_pts.flatten())
        
        if new_val < current_val:
            current_x = new_x
            current_val = new_val
            if new_val < best_val:
                best_x = new_x
                best_val = new_val
        else:
            prob = np.exp((current_val - new_val) / T)
            if rng.random() < prob:
                current_x = new_x
                current_val = new_val
        
        T *= cooling
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = None
    best_val = np.inf
    
    # Phase 1: L-BFGS-B on smooth surrogate from symmetric start
    x0 = _symmetric_initial_config()
    x, val = _lbfgsb_optimize(x0, max_time=10.0)
    best_x, best_val = x, val
    
    # SLSQP polish
    if time.time() < deadline - 30:
        x, val = _slsqp_polish(best_x, max_time=5.0)
        if val < best_val:
            best_x, best_val = x, val
    
    # Phase 2: Multi-start with L-BFGS-B
    for seed in range(40):
        if time.time() > deadline - 25:
            break
        
        if seed % 3 == 0:
            x0 = _random_initial_config(seed + 100)
        elif seed % 3 == 1:
            x0 = _generate_structured_start(seed + 500)
        else:
            x0 = _symmetric_initial_config()
            # Perturb slightly
            rng = np.random.RandomState(seed)
            x0 = x0 + rng.normal(0, 0.05, size=N*2)
            x0 = _project_to_triangle(x0.reshape(N, 2)).flatten()
        
        x, val = _lbfgsb_optimize(x0, max_time=5.0)
        
        # Quick SLSQP polish for promising candidates
        if val < best_val + 0.001:
            x, val = _slsqp_polish(x, max_time=3.0)
        
        if val < best_val:
            best_x, best_val = x, val
    
    # Phase 3: Simulated annealing from best
    if time.time() < deadline - 15:
        sa_time = min(5.0, deadline - time.time() - 10)
        if sa_time > 1.0:
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            sa_x, sa_val = _simulated_annealing(best_x, max_time=sa_time, rng=rng)
            if sa_val < best_val:
                best_x, best_val = sa_x, sa_val
    
    # Final high-precision polish
    if time.time() < deadline - 8:
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, _H))
        try:
            res = minimize(
                _true_objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
