"""Heilbronn triangles: maximize the smallest triangle area among 11 points.

Combines:
- Barycentric coordinate representation (natural constraint enforcement)
- Soft-min (log-sum-exp) differentiable objective for gradient-based search
- Push-apart heuristic
- Coordinate descent on hard min
- Nelder-Mead polish on active set
"""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 140.0


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
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


def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _soft_min_area(pts, beta):
    """Differentiable soft minimum of triangle areas via log-sum-exp."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _push_apart(pts, alpha=0.02, max_iter=50):
    """Iteratively push all points apart to increase minimum area."""
    pts = pts.copy()
    current_min = _min_triangle_area(pts)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            best_pos = pts[i].copy()
            best_val = current_min
            
            # Compute force from all other points
            force = np.zeros(2)
            for j in range(N):
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    force += d / (dist * dist)
            
            # Try moving in force direction
            norm_f = np.linalg.norm(force)
            if norm_f > 1e-10:
                direction = force / norm_f
                for step in [0.005, 0.01, 0.02, 0.05]:
                    new_pos = pts[i] + step * direction
                    new_pos = _project_to_triangle(new_pos.reshape(1, 2))[0]
                    new_pts = pts.copy()
                    new_pts[i] = new_pos
                    new_min = _min_triangle_area(new_pts)
                    if new_min > best_val + 1e-12:
                        best_val = new_min
                        best_pos = new_pos
            
            if np.linalg.norm(best_pos - pts[i]) > 1e-12:
                pts[i] = best_pos
                current_min = _min_triangle_area(pts)
                improved = True
        
        if not improved:
            break
    
    return pts


def _coordinate_descent(pts, max_iter=100, tol=1e-10):
    """Coordinate descent to maximize the minimum triangle area."""
    current_min = _min_triangle_area(pts)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        improved = False
        for idx in range(N):
            best_local = current_min
            best_pos = pts[idx].copy()
            
            for step in step_sizes:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] = best_pos + np.array([dx, dy])
                        new_pts = _project_to_triangle(new_pts)
                        new_min = _min_triangle_area(new_pts)
                        if new_min > best_local + tol:
                            best_local = new_min
                            best_pos = new_pts[idx].copy()
            
            if np.linalg.norm(best_pos - pts[idx]) > 1e-12:
                pts[idx] = best_pos
                current_min = _min_triangle_area(pts)
                improved = True
        
        if not improved:
            break
    
    return pts


def _active_set(pts, tol=1.02):
    """Return indices of triangles with area <= tol * min_area."""
    areas = _all_triangle_areas(pts)
    min_a = np.min(areas)
    return np.where(areas <= tol * min_a)[0]


def _nm_polish(pts, maxiter=5000, xatol=1e-12, fatol=1e-14):
    """Nelder-Mead polish on the hard min objective."""
    def obj(x):
        px = _project_to_triangle(x.reshape(-1, 2))
        return -_min_triangle_area(px)
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
        return _project_to_triangle(res.x.reshape(-1, 2))
    except Exception:
        return pts


def _soft_min_nm(pts, beta, maxiter=3000):
    """Nelder-Mead polish using soft-min objective (differentiable)."""
    def obj(x):
        px = _project_to_triangle(x.reshape(-1, 2))
        return -_soft_min_area(px, beta)
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
        return _project_to_triangle(res.x.reshape(-1, 2))
    except Exception:
        return pts


def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = v * S3H
    return np.array([x, y])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Structured initial configurations
    inits = []
    
    # Config 1: Corners + edge midpoints + interior
    init1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    inits.append(init1)
    
    # Config 2: More symmetric interior placement
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H * 0.5],
        [0.25, S3H * 0.5],
        [0.75, S3H * 0.5],
        [0.5, S3H * 0.25],
        [0.5, S3H * 0.75],
        [0.5, S3H * 0.45],
    ], dtype=float)
    inits.append(init2)
    
    # Config 3: Based on 3x3 grid + corners
    init3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2],
        [0.333, S3H/3],
        [0.667, S3H/3],
    ], dtype=float)
    inits.append(init3)
    
    for init in inits:
        _update_best(init)
    
    # Phase A: Multi-start with push-apart + coordinate descent + soft-min NM
    n_starts = 30
    bases = list(inits)
    for _ in range(15):
        base_idx = rng.integers(0, len(inits))
        base = inits[base_idx] + rng.normal(0, 0.2, size=(N, 2))
        bases.append(_project_to_triangle(base))
    for _ in range(10):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
    
    for s in range(n_starts):
        if not _elapsed_ok(45):
            break
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.03, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)
        
        # Apply push-apart
        try:
            pts = _push_apart(pts, alpha=0.02, max_iter=30)
            _update_best(pts)
        except Exception:
            pass
        
        # Apply coordinate descent
        try:
            pts = _coordinate_descent(pts, max_iter=80)
            _update_best(pts)
        except Exception:
            pass
        
        # Soft-min NM polish (differentiable, helps escape local minima)
        if _elapsed_ok(35):
            try:
                for beta in [20, 50, 100]:
                    pts = _soft_min_nm(pts, beta, maxiter=2000)
                    _update_best(pts)
            except Exception:
                pass
    
    # Phase B: Nelder-Mead polish with hard min on best candidates
    if best_pts is not None and _elapsed_ok(30):
        bc_best = best_pts.copy()
        
        for scale in [0.03, 0.015, 0.008, 0.004]:
            if not _elapsed_ok(20):
                break
            for _ in range(6):
                if not _elapsed_ok(15):
                    break
                pert = bc_best + rng.normal(0, scale, size=(N, 2))
                pert = _project_to_triangle(pert)
                
                try:
                    p = _nm_polish(pert, maxiter=5000, xatol=1e-12, fatol=1e-14)
                    _update_best(p)
                    if _min_triangle_area(p) > _min_triangle_area(bc_best):
                        bc_best = p.copy()
                except Exception:
                    pass
    
    # Phase C: Active-set coordinate descent + ultra-fine NM
    if best_pts is not None and _elapsed_ok(15):
        pts = best_pts.copy()
        for _ in range(30):
            if not _elapsed_ok(10):
                break
            areas = _all_triangle_areas(pts)
            min_a = np.min(areas)
            active = set()
            for i, a in enumerate(areas):
                if a < min_a * 1.01:
                    active.add(_T_I[i])
                    active.add(_T_J[i])
                    active.add(_T_K[i])
            
            improved = False
            for pt_idx in active:
                for _ in range(3):
                    if not _elapsed_ok(8):
                        break
                    new_pts = pts.copy()
                    new_pts[pt_idx] += rng.normal(0, 0.005, size=2)
                    new_pts = _project_to_triangle(new_pts)
                    if _min_triangle_area(new_pts) > min_a:
                        pts = new_pts
                        improved = True
                        min_a = _min_triangle_area(pts)
                        break
            if not improved:
                break
        _update_best(pts)
        
        # Final ultra-fine NM
        if _elapsed_ok(8):
            try:
                p = _nm_polish(best_pts, maxiter=10000, xatol=1e-14, fatol=1e-16)
                _update_best(p)
            except Exception:
                pass
        
        # Final coordinate descent
        if _elapsed_ok(5):
            try:
                p = _coordinate_descent(best_pts, max_iter=150)
                _update_best(p)
            except Exception:
                pass
    
    if best_pts is None:
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        return pts
    
    return _project_to_triangle(best_pts)
