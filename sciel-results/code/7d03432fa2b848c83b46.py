"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 centrally symmetric points.
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    """Negative score for minimization (symmetric params)."""
    points = _params_to_points(params)
    score = _score(points)
    return -score

def _objective_smooth(params, beta=50.0):
    """
    Smooth surrogate for symmetric params.
    """
    points = _params_to_points(params)
    return _objective_smooth_coords(points, beta)

def _objective_smooth_coords(points, beta=50.0):
    """
    Smooth surrogate for 28-dim coordinate vector.
    Uses log-sum-exp for soft-min of triangle areas.
    """
    pts = points.reshape(N, 2)
    areas = _all_areas(pts)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _bounds_sym():
    bounds = []
    for _ in range(N_HALF):
        bounds.append((0.02, np.pi - 0.02))
    for _ in range(N_HALF):
        bounds.append((0.05, 3.0))
    return bounds

def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain (faster for small N)."""
    pts = np.array(pts, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])

def construct():
    """
    Construct 14 points maximizing min triangle area / convex hull area.
    Strategy:
    1. Global search using Differential Evolution on symmetric parameters.
    2. Local refinement using Smooth Surrogate (soft-min) on symmetric parameters.
    3. Final high-precision polish using Direct Coordinate Optimization (28-dim) with Smooth Surrogate.
    4. Jitter-and-search to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = 0.0
    best_params = None
    rng = np.random.default_rng(42)
    
    bounds = _bounds_sym()
    
    # --- Phase 1: Global Exploration (DE) ---
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol',
                mutation=(0.4, 1.0),
                recombination=0.7
            )
            if np.isfinite(de_result.fun):
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_params = de_result.x.copy()
                    best_points = _params_to_points(de_result.x)
        except Exception:
            pass

    # --- Phase 2: Local Refinement on Symmetric Parameters ---
    # Generate diverse starting points
    inits = []
    if best_params is not None:
        inits.append(best_params)
    
    # Regular polygon
    base_angles = np.linspace(0.1, np.pi - 0.1, N_HALF)
    base_radii = np.ones(N_HALF)
    inits.append(np.concatenate([base_angles, base_radii]))
    
    # Random starts
    for _ in range(20):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.5, 1.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))

    # Refine top candidates using smooth surrogate
    scored_inits = []
    for init_params in inits:
        if time.time() > deadline - 40:
            break
        try:
            s = -_objective_smooth(init_params, beta=50.0)
            scored_inits.append((s, init_params))
        except Exception:
            pass
            
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    candidates = [p for _, p in scored_inits[:5]]
    
    for init_params in candidates:
        if time.time() > deadline - 10:
            break
        
        # 1. Smooth optimization
        try:
            res_smooth = minimize(
                lambda p: _objective_smooth(p, beta=50.0), 
                init_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if np.isfinite(res_smooth.fun):
                # Evaluate true score
                true_score = -_objective(res_smooth.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = res_smooth.x.copy()
                    best_points = _params_to_points(res_smooth.x)
        except Exception:
            pass
        
        # 2. Raw objective refinement from the best symmetric point found so far
        if best_params is not None:
            try:
                res_raw = minimize(
                    _objective, 
                    best_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if np.isfinite(res_raw.fun):
                    score = -res_raw.fun
                    if score > best_score:
                        best_score = score
                        best_params = res_raw.x.copy()
                        best_points = _params_to_points(res_raw.x)
            except Exception:
                pass

    # --- Phase 3: Direct Coordinate Optimization (Break Symmetry) ---
    # The optimal configuration for n=14 might not be perfectly centrally symmetric.
    # We now optimize the 28 coordinates directly using the smooth surrogate.
    
    if best_points is not None:
        # Start from the best symmetric solution
        x0 = best_points.flatten()
        
        # Try multiple perturbations around the best point
        for trial in range(15):
            if time.time() > deadline - 5:
                break
            
            # Jitter
            if trial == 0:
                x_start = x0
            else:
                x_start = x0 + rng.normal(0, 0.1, N * 2)
            
            try:
                # Optimize with smooth surrogate
                res_direct = minimize(
                    lambda x: _objective_smooth_coords(x, beta=80.0), 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if np.isfinite(res_direct.fun):
                    pts_candidate = res_direct.x.reshape(N, 2)
                    # Check if it's valid (non-degenerate)
                    hull_area = _compute_hull_area(pts_candidate)
                    if hull_area > 1e-6:
                        score = _score(pts_candidate)
                        if score > best_score:
                            best_score = score
                            best_points = pts_candidate.copy()
            except Exception:
                pass

    # --- Phase 4: Final High-Precision Polish ---
    if best_points is not None and time.time() < deadline - 2:
        # Very tight Nelder-Mead on the raw objective (non-smooth but precise)
        x_best = best_points.flatten()
        try:
            res_final = minimize(
                lambda x: -_score(x.reshape(N, 2)), 
                x_best, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if np.isfinite(res_final.fun):
                pts_final = res_final.x.reshape(N, 2)
                score_final = _score(pts_final)
                if score_final > best_score:
                    best_score = score_final
                    best_points = pts_final.copy()
        except Exception:
            pass
        
        # Coordinate-wise fine search (Hill Climbing)
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(N):
                    for dim in range(2):
                        for delta in [-0.001, -0.0005, 0.0005, 0.001]:
                            candidate = best_points.copy()
                            candidate[i, dim] += delta
                            s = _score(candidate)
                            if s > best_score + 1e-12:
                                best_score = s
                                best_points = candidate.copy()
                                improved = True

    if best_points is None:
        theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])

    return best_points
