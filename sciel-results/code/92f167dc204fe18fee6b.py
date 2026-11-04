"""Heilbronn convex regions n=14: maximize minimum triangle area over convex hull area."""
from scipy.optimize import minimize
import time
import numpy as np

N = 14
N_HEPT = 7  # Base heptagon vertices, each replaced by a pair

# Precompute triple indices for vectorized area computation
_TRIPLE_IDX = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _TRIPLE_IDX.append((_i, _j, _k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX)  # shape (364, 3)

def _compute_all_areas(points):
    """Vectorized computation of all 364 triangle areas. Returns array of shape (364,)."""
    # points: (14, 2)
    # For each triple (i,j,k): area = 0.5 * |(p_j - p_i) x (p_k - p_i)|
    pi = points[_TRIPLE_IDX[:, 0]]  # (364, 2)
    pj = points[_TRIPLE_IDX[:, 1]]  # (364, 2)
    pk = points[_TRIPLE_IDX[:, 2]]  # (364, 2)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _convex_hull_area_fast(points):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: shoelace on angular sort
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        sp = points[order]
        x, y = sp[:, 0], sp[:, 1]
        return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

def _score_from_points(points):
    """Score = min triangle area / convex hull area."""
    areas = _compute_all_areas(points)
    min_a = np.min(areas)
    hull_a = _convex_hull_area_fast(points)
    if hull_a < 1e-12:
        return 0.0
    return min_a / hull_a

def _params_to_points(params):
    """
    Convert 10 parameters to 14 points.
    params[0:7]  : radial offsets for the 7 heptagon vertices (pair separation along radial direction)
    params[7]    : overall scale of the heptagon
    params[8]    : rotation angle
    params[9]    : shear parameter (affine transform)
    
    Representation:
      - Start with regular heptagon of radius R = params[7]
      - Each vertex v_i gets a pair: v_i + d_i * radial_dir_i and v_i - d_i * radial_dir_i
      - Apply rotation and shear to all 14 points
    """
    d = params[0:7]       # radial offsets (can be negative)
    R = params[7]         # base radius
    theta0 = params[8]    # rotation
    shear = params[9]     # affine shear
    
    # Heptagon angles
    base_angles = 2.0 * np.pi * np.arange(N_HEPT) / N_HEPT + theta0
    
    # Radial directions
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)
    
    # For each heptagon vertex, create two points offset along radial direction
    # Point pair: (R + d_i) * (cos, sin) and (R - d_i) * (cos, sin)
    r_outer = R + d
    r_inner = R - d
    
    x_outer = r_outer * cos_a
    y_outer = r_outer * sin_a
    x_inner = r_inner * cos_a
    y_inner = r_inner * sin_a
    
    # Interleave: for each heptagon vertex, place outer then inner
    points = np.empty((2 * N_HEPT, 2))
    points[0::2, 0] = x_outer
    points[0::2, 1] = y_outer
    points[1::2, 0] = x_inner
    points[1::2, 1] = y_inner
    
    # Apply affine shear: x' = x + shear * y, y' = y
    points[:, 0] = points[:, 0] + shear * points[:, 1]
    
    return points

def _negative_score(params):
    """Objective for minimization: negative of score."""
    pts = _params_to_points(params)
    return -_score_from_points(pts)

def _soft_min_score(params, temperature=0.01):
    """
    Smooth approximation of min area using log-sum-exp for gradient-friendly optimization.
    Returns -soft_min / hull_area (to minimize).
    """
    pts = _params_to_points(params)
    areas = _compute_all_areas(pts)
    # Soft min via log-sum-exp: soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum
    hull_a = _convex_hull_area_fast(pts)
    if hull_a < 1e-12:
        return 0.0
    return -soft_min / hull_a

def _optimize_params(params0, time_budget, method='L-BFGS-B', use_soft=False, temperature=0.01):
    """Optimize the 10-parameter representation."""
    start = time.time()
    
    if use_soft:
        def obj(x):
            if time.time() - start > time_budget:
                return 0.0
            return _soft_min_score(x, temperature)
    else:
        def obj(x):
            if time.time() - start > time_budget:
                return 0.0
            return _negative_score(x)
    
    bounds = [(-0.5, 0.5)] * 7 + [(0.3, 3.0), (-np.pi, np.pi), (-0.5, 0.5)]
    
    try:
        result = minimize(obj, params0, method=method, bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - start > time_budget:
            return params0
        return result.x
    except Exception:
        return params0

def construct():
    """Construct 14 points using heptagon-pair representation with affine transform."""
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Initial parameters: symmetric configuration
    # d = 0 (all points on heptagon), R = 1, rotation = 0, shear = 0
    init_params = np.zeros(10)
    init_params[7] = 1.0  # R = 1
    
    # --- Phase 1: Optimize with exact min (Nelder-Mead for non-smooth) ---
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        t_budget = min(40.0, remaining - 5)
        opt_params = _optimize_params(init_params, t_budget, method='Nelder-Mead')
        pts = _params_to_points(opt_params)
        s = _score_from_points(pts)
        if s > best_score:
            best_score = s
            best_params = opt_params.copy()
            best_points = pts.copy()
    
    # --- Phase 2: Polish with L-BFGS-B on soft-min (smooth surrogate) ---
    remaining = deadline - (time.time() - start_time)
    if remaining > 10 and best_params is not None:
        t_budget = min(30.0, remaining - 5)
        # Use soft-min with decreasing temperature
        for temp in [0.05, 0.01, 0.001]:
            if time.time() - start_time > deadline - 10:
                break
            sub_budget = min(10.0, deadline - (time.time() - start_time) - 5)
            if sub_budget < 3:
                break
            opt_params = _optimize_params(best_params, sub_budget, method='L-BFGS-B', use_soft=True, temperature=temp)
            pts = _params_to_points(opt_params)
            s = _score_from_points(pts)
            if s > best_score:
                best_score = s
                best_params = opt_params.copy()
                best_points = pts.copy()
    
    # --- Phase 3: Final exact polish with Nelder-Mead on best params ---
    remaining = deadline - (time.time() - start_time)
    if remaining > 10 and best_params is not None:
        t_budget = min(20.0, remaining - 5)
        opt_params = _optimize_params(best_params, t_budget, method='Nelder-Mead')
        pts = _params_to_points(opt_params)
        s = _score_from_points(pts)
        if s > best_score:
            best_score = s
            best_params = opt_params.copy()
            best_points = pts.copy()
    
    # --- Phase 4: Try a few alternative initial conditions ---
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        rng = np.random.RandomState(42)
        n_trials = min(4, int(remaining / 8))
        for trial in range(n_trials):
            if time.time() - start_time > deadline - 10:
                break
            # Perturb initial params
            p_try = init_params.copy()
            p_try[0:7] = rng.normal(0, 0.1, 7)
            p_try[7] = 1.0 + rng.normal(0, 0.1)
            p_try[8] = rng.normal(0, 0.3)
            p_try[9] = rng.normal(0, 0.1)
            
            t_budget = min(15.0, deadline - (time.time() - start_time) - 5)
            if t_budget < 3:
                break
            opt_params = _optimize_params(p_try, t_budget, method='Nelder-Mead')
            pts = _params_to_points(opt_params)
            s = _score_from_points(pts)
            if s > best_score:
                best_score = s
                best_params = opt_params.copy()
                best_points = pts.copy()
    
    # Fallback: if optimization failed, use regular 14-gon
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
