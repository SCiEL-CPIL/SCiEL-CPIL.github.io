"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

# Precompute all C(11,3) = 165 triangle index triplets
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

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _clamp_to_triangle(points):
    """Clamp points to be inside the unit equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    y = np.maximum(y, 0)
    # y <= sqrt(3)*x
    y = np.minimum(y, np.sqrt(3) * x)
    # y <= sqrt(3)*(1-x)
    y = np.minimum(y, np.sqrt(3) * (1 - x))
    # x in [0, 1]
    x = np.clip(x, 0, 1)
    # If x is very close to 0 or 1, y must be 0
    x[x < 1e-10] = 0
    x[x > 1 - 1e-10] = 1
    y[np.abs(x) < 1e-10] = 0
    y[np.abs(x - 1) < 1e-10] = 0
    y = np.clip(y, 0, H)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _objective(params):
    """Objective function: -min_area + penalty. Minimize this."""
    points = params.reshape(N, 2)
    pen = _penalty(points)
    if pen > 1.0:
        return 1e6 + pen
    area = _min_area(points)
    return -area + 10.0 * pen

def _soft_min_objective(params, beta=100.0):
    """Smooth objective using log-sum-exp soft minimum of areas.
    Returns negative soft-min area + penalty."""
    points = params.reshape(N, 2)
    pen = _penalty(points)
    if pen > 1.0:
        return 1e6 + pen
    areas = _all_areas(points)
    # Soft minimum: -log(sum(exp(-beta * areas))) / beta
    # This approximates min(areas) as beta -> infinity
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    soft_min = log_sum / beta
    return -soft_min + 10.0 * pen

def _exact_objective(params):
    """Exact objective: -min_area + penalty."""
    return _objective(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Bounds for x and y
    bounds = [(0, 1), (0, H)] * N
    
    best_params = None
    best_score = -np.inf
    
    # Helper to update best
    def update_best(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Phase 1: Run DE with soft-min objective for better global exploration
    # The soft-min is smooth, allowing DE to explore more effectively
    de_deadline = start_time + 60
    try:
        def _de_obj_soft(params):
            if time.time() > de_deadline:
                return 1e9
            return _soft_min_objective(params, beta=100.0)
        
        res_de = differential_evolution(
            _de_obj_soft,
            bounds,
            maxiter=150,
            popsize=35,
            tol=1e-7,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        if res_de.x is not None:
            pts = _clamp_to_triangle(res_de.x.reshape(N, 2))
            score = _min_area(pts)
            update_best(pts, score)
    except Exception:
        pass
    
    # Phase 2: Run another DE with slightly different settings and exact objective
    if time.time() < deadline - 40:
        try:
            res_de2 = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=25,
                tol=1e-6,
                seed=123,
                polish=False,
                init='latinhypercube'
            )
            if res_de2.x is not None:
                pts = _clamp_to_triangle(res_de2.x.reshape(N, 2))
                score = _min_area(pts)
                update_best(pts, score)
        except Exception:
            pass
    
    # Phase 3: Polish the best candidate with Nelder-Mead on exact objective
    if best_params is not None and time.time() < deadline - 15:
        try:
            params_flat = best_params.flatten()
            res_nm = minimize(
                _exact_objective,
                params_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.x is not None:
                pts = _clamp_to_triangle(res_nm.x.reshape(N, 2))
                score = _min_area(pts)
                update_best(pts, score)
        except Exception:
            pass
        
        # Try L-BFGS-B as well
        if time.time() < deadline - 10:
            try:
                params_flat = best_params.flatten()
                res_lbfgs = minimize(
                    _exact_objective,
                    params_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500}
                )
                if res_lbfgs.x is not None:
                    pts = _clamp_to_triangle(res_lbfgs.x.reshape(N, 2))
                    score = _min_area(pts)
                    update_best(pts, score)
            except Exception:
                pass
    
    # Phase 4: Multi-start local polish with random restarts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(456)
        for i in range(8):
            if time.time() > deadline - 8:
                break
            
            # Generate random points inside triangle
            pts = np.zeros((N, 2))
            for j in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v < 1:
                        pts[j, 0] = v + 0.5 * (1 - u - v)
                        pts[j, 1] = H * (1 - u - v)
                        break
            
            # Polish with Nelder-Mead
            try:
                res_r = minimize(
                    _exact_objective,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_r.x is not None:
                    pts_r = _clamp_to_triangle(res_r.x.reshape(N, 2))
                    score = _min_area(pts_r)
                    update_best(pts_r, score)
            except Exception:
                pass
    
    # Phase 5: If we still have time, try more restarts with different random seeds
    if time.time() < deadline - 10:
        rng2 = np.random.default_rng(789)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            # Start from a perturbation of current best
            if best_params is not None:
                pts = best_params + rng2.normal(0, 0.02, size=(N, 2))
                pts = _clamp_to_triangle(pts)
            else:
                pts = np.zeros((N, 2))
                for j in range(N):
                    while True:
                        u = rng2.random()
                        v = rng2.random()
                        if u + v < 1:
                            pts[j, 0] = v + 0.5 * (1 - u - v)
                            pts[j, 1] = H * (1 - u - v)
                            break
            
            try:
                res_r = minimize(
                    _exact_objective,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res_r.x is not None:
                    pts_r = _clamp_to_triangle(res_r.x.reshape(N, 2))
                    score = _min_area(pts_r)
                    update_best(pts_r, score)
            except Exception:
                pass
    
    # Fallback if no valid result
    if best_params is None:
        pts = np.zeros((N, 2))
        rng = np.random.default_rng(0)
        for j in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    pts[j, 0] = v + 0.5 * (1 - u - v)
                    pts[j, 1] = H * (1 - u - v)
                    break
        best_params = _clamp_to_triangle(pts)
    
    return _clamp_to_triangle(best_params)
