# EVOLVE-BLOCK-START
import numpy as np
import time
from math import pi, cos, sin, ceil

# Deadline guard: keep a safety margin before the 630s kill.
_START = time.time()
DEADLINE = _START + 615.0  # 15s safety margin


def _is_inside_base(u, v):
    """
    Vectorized check: are points (u, v) inside the base L-shaped corridor?
    Base = {0<=u<=1, -3<=v<=1} ∪ {-3<=u<=1, 0<=v<=1}
    """
    # Rect 1: 0 <= u <= 1, -3 <= v <= 1
    r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Rect 2: -3 <= u <= 1, 0 <= v <= 1
    r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return r1 | r2


def _build_full_path(path):
    """
    Prepend (0,0,0) and append (-2,0,90), then densify.
    Returns array of shape (m, 3).
    """
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    pts = np.vstack([start, path, end])

    # Densify: between consecutive rows, add linear interpolations in m = max(1, ceil(|a2-a1|)) equal steps
    full = [pts[0]]
    for i in range(len(pts) - 1):
        a1 = pts[i, 2]
        a2 = pts[i + 1, 2]
        m = max(1, int(ceil(abs(a2 - a1))))
        # m equal steps means m intervals, so m+1 points including endpoints
        # But we already have the start of this segment; add m points (the end is next segment's start)
        for k in range(1, m + 1):
            t = k / m
            x = pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])
            y = pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1])
            a = a1 + t * (a2 - a1)
            full.append(np.array([x, y, a]))
    full = np.array(full, dtype=np.float64)
    return full


def _sofa_area_exact(path):
    """
    Compute the exact sofa area (number of grid points * 0.0001) for a given path.
    This is the true objective.
    """
    full = _build_full_path(path)
    # Sofa grid
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    npts = len(pts)

    # For each placement, check which sofa points are inside
    # A sofa point must be inside EVERY placement
    inside_all = np.ones(npts, dtype=bool)

    for (x, y, a_deg) in full:
        a = np.radians(a_deg)
        ca, sa = cos(a), sin(a)
        # R(-a) applied to (w - (x,y)): 
        # dx = w_x - x, dy = w_y - y
        # u = ca*dx + sa*dy
        # v = -sa*dx + ca*dy
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside = _is_inside_base(u, v)
        inside_all &= inside
        if not inside_all.any():
            break  # no points remain, area is 0

    return inside_all.sum() * 0.0001


def _sofa_area_soft(path, beta=100.0):
    """
    Soft surrogate for the sofa area. Instead of requiring a point to be inside ALL placements,
    we use a soft penalty: for each point, compute the minimum 'distance' to being inside across placements,
    and use a log-sum-exp style soft-min to approximate the hard min.
    
    Actually, a better approach: for each point, compute a 'violation' score = max over placements of 
    (distance outside the base), then the soft area is sum over points of exp(-beta * violation).
    This is smooth in the path parameters.
    """
    full = _build_full_path(path)
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    npts = len(pts)

    # For each point, track the maximum violation across all placements
    # Violation for a placement: 0 if inside, else distance to the boundary (approximate)
    max_violation = np.zeros(npts, dtype=np.float64)

    for (x, y, a_deg) in full:
        a = np.radians(a_deg)
        ca, sa = cos(a), sin(a)
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy

        # Distance to base L-shape: if inside, 0; else min distance to the two rects
        # Rect1: [0,1] x [-3,1]
        du1 = np.maximum(0, np.maximum(0 - u, u - 1))
        dv1 = np.maximum(0, np.maximum(-3 - v, v - 1))
        d1 = np.sqrt(du1**2 + dv1**2)
        # Rect2: [-3,1] x [0,1]
        du2 = np.maximum(0, np.maximum(-3 - u, u - 1))
        dv2 = np.maximum(0, np.maximum(0 - v, v - 1))
        d2 = np.sqrt(du2**2 + dv2**2)
        dist = np.minimum(d1, d2)

        max_violation = np.maximum(max_violation, dist)

    # Soft area: sum of exp(-beta * max_violation) * 0.0001
    # This is smooth and approximates the hard count as beta -> inf
    soft_count = np.sum(np.exp(-beta * max_violation))
    return soft_count * 0.0001


def _nelder_mead(objective, x0, maxiter=200, tol=1e-6, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """
    Simple Nelder-Mead implementation.
    """
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if x0[i] == 0 else 0.1 * x0[i]
        simplex.append(xi)
    
    values = [objective(x) for x in simplex]
    
    for it in range(maxiter):
        # Sort simplex by objective value (maximize)
        order = np.argsort(values)[::-1]
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[0] - values[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr >= values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe > fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr >= values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc > values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    best_idx = np.argmax(values)
    return simplex[best_idx], values[best_idx]


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path to maximize the sofa area.
    """
    # Initial path from the baseline
    path0 = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    best_path = path0.copy()
    best_score = _sofa_area_soft(path0, beta=50.0)

    # Phase 1: Coarse soft optimization
    beta_coarse = 20.0
    obj_coarse = lambda x: _sofa_area_soft(x.reshape(-1, 3), beta=beta_coarse)
    
    x0 = path0.ravel()
    # Run Nelder-Mead on the flattened path
    x_best, v_best = _nelder_mead(obj_coarse, x0, maxiter=150, tol=1e-4)
    path_c = x_best.reshape(-1, 3)
    score_c = _sofa_area_soft(path_c, beta=beta_coarse)
    if score_c > best_score:
        best_path = path_c
        best_score = score_c

    if time.time() > DEADLINE - 120:
        # Not enough time for more phases
        return _sofa_area_exact(best_path) and best_path or best_path

    # Phase 2: Medium soft optimization
    beta_med = 100.0
    obj_med = lambda x: _sofa_area_soft(x.reshape(-1, 3), beta=beta_med)
    x_best, v_best = _nelder_mead(obj_med, best_path.ravel(), maxiter=150, tol=1e-5)
    path_m = x_best.reshape(-1, 3)
    score_m = _sofa_area_soft(path_m, beta=beta_med)
    if score_m > _sofa_area_soft(best_path, beta=beta_med):
        best_path = path_m

    if time.time() > DEADLINE - 60:
        return best_path

    # Phase 3: Exact local polish
    obj_exact = lambda x: _sofa_area_exact(x.reshape(-1, 3))
    x_best, v_best = _nelder_mead(obj_exact, best_path.ravel(), maxiter=80, tol=1e-7)
    path_e = x_best.reshape(-1, 3)
    score_e = _sofa_area_exact(path_e)
    score_best_exact = _sofa_area_exact(best_path)
    if score_e > score_best_exact:
        best_path = path_e

    return best_path


# EVOLVE-BLOCK-END
