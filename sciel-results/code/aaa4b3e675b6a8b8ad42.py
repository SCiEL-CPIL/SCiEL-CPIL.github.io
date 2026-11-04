"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _softmin_objective(params, beta=50.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _softmin_grad(params, beta=50.0):
    """Gradient of the soft-min objective."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(area_m) = -exp_term[m] / sum_exp
    d_softmin_d_area = -exp_term / sum_exp
    
    # d(area_m)/d(point)
    # area = 0.5 * |cross|
    # sign = sign(cross)
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    # d(cross)/d(pi) = (-(dy_k), dx_k)
    # d(cross)/d(pj) = (dy_k, -dx_k)
    # d(cross)/d(pk) = (-dy_i, dx_i)
    
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_d_pi = np.column_stack([-dy_k, dx_k])
    contrib_i = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pi
    np.add.at(grad_points, _IDX_I, contrib_i)
    
    # Contribution from pj
    d_cross_d_pj = np.column_stack([dy_k, -dx_k])
    contrib_j = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pj
    np.add.at(grad_points, _IDX_J, contrib_j)
    
    # Contribution from pk
    d_cross_d_pk = np.column_stack([-dy_i, dx_i])
    contrib_k = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pk
    np.add.at(grad_points, _IDX_K, contrib_k)
    
    # Now map grad_points to grad_params
    # points = [x, y]
    # x = s(1-t)*V1[0] + s*t*V2[0]
    # y = s(1-t)*V1[1] + s*t*V2[1]
    
    s = params[0::2]
    t = params[1::2]
    
    # dx/ds = (1-t)*V1[0] + t*V2[0]
    # dx/dt = -s*V1[0] + s*V2[0]
    # dy/ds = (1-t)*V1[1] + t*V2[1]
    # dy/dt = -s*V1[1] + s*V2[1]
    
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_s = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_t = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_s
    grad_params[1::2] = grad_t
    
    # Objective is -soft_min, so gradient is -grad_softmin
    # We want to minimize -soft_min, so we return -grad_points mapped to params
    # Wait, _softmin_objective returns -soft_min.
    # d(-soft_min)/d(params) = - d(soft_min)/d(params)
    # My calculation above was for d(soft_min)/d(area) -> d(soft_min)/d(points) -> d(soft_min)/d(params)
    # So the gradient of the objective (which is -soft_min) is the negative of that.
    
    return -grad_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 10
    for i in range(n_fps_inits):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(10):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP on the exact objective
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Polish the best solution found with Nelder-Mead on the exact objective
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 4. Try a few more SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # 5. Trust-Region Reflective on the smooth surrogate (log-sum-exp)
    if best_params is not None and time_left() > 20.0:
        try:
            res = minimize(
                _softmin_objective, 
                best_params, 
                jac=_softmin_grad,
                method='TRF', 
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            # Evaluate the EXACT objective at the result of the surrogate optimization
            exact_score = -_objective(res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. Final Nelder-Mead polish on the exact objective
    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Final fallback
    if best_params is None:
        pts = _generate_fps_initialization(rng)
        best_params = _points_to_param(pts)
    
    best_points = _param_to_points(best_params)
    return np.array(best_points)
