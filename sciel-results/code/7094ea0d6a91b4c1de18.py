"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

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
_N_TRIS = len(_idx_i)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    return np.min(_all_areas(points))

def _inside_penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, H)
        max_y_left = np.sqrt(3) * x
        if y > max_y_left:
            y = max_y_left
        max_y_right = np.sqrt(3) * (1 - x)
        if y > max_y_right:
            y = max_y_right
        if x < 1e-12:
            y = 0.0
        if x > 1 - 1e-12:
            y = 0.0
        pts[i] = [x, y]
    return pts

def _params_to_points_symmetric(params):
    """
    Map symmetric parameters to 11 points with reflection symmetry about x=0.5.
    
    Structure:
    - Point 0: on the symmetry axis (x=0.5), parameterized by y only -> 1 DOF
    - Points 1-10: 5 pairs of mirrored points. Each pair (i, i+5) where:
      - Point i: (x, y) with x in (0, 0.5), y >= 0
      - Point i+5: (1-x, y)
    
    So total DOF = 1 (axis point y) + 5 * 2 (x,y for each left-side point) = 11 DOF
    
    params: array of shape (11,)
      params[0] = y of the axis point
      params[1:6] = (x1, y1) for pair 1
      params[6:11] = ... wait, that's 1 + 5*2 = 11, but I have 11 params.
      
    Let me redo: 
      params[0] = y of center point (x=0.5)
      params[1], params[2] = x, y of left point in pair 1
      params[3], params[4] = x, y of left point in pair 2
      params[5], params[6] = x, y of left point in pair 3
      params[7], params[8] = x, y of left point in pair 4
      params[9], params[10] = x, y of left point in pair 5
    
    Total: 1 + 10 = 11 params. Good.
    """
    pts = np.zeros((N, 2))
    # Center point on axis
    pts[0] = [0.5, params[0]]
    
    # 5 pairs
    for i in range(5):
        x = params[1 + 2*i]
        y = params[2 + 2*i]
        # Left point
        pts[1 + 2*i] = [x, y]
        # Right mirror
        pts[2 + 2*i] = [1.0 - x, y]
    
    return pts

def _points_to_params_symmetric(points):
    """Extract symmetric parameters from a full point set (for initialization)."""
    pts = points.copy()
    params = np.zeros(11)
    # Center point: find the one closest to x=0.5
    # For now, assume point 0 is the center
    params[0] = pts[0, 1]
    # For pairs, take the left one (smaller x)
    for i in range(5):
        left_idx = 1 + 2*i
        right_idx = 2 + 2*i
        # Average y, take min x for the left
        x_left = min(pts[left_idx, 0], pts[right_idx, 0])
        y_avg = (pts[left_idx, 1] + pts[right_idx, 1]) / 2.0
        params[1 + 2*i] = x_left
        params[2 + 2*i] = y_avg
    return params

def _objective_symmetric(params):
    """Objective in symmetric parameter space: -min_area + penalty."""
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 0.1:
        return 1e6 + pen
    area = _min_area(pts)
    return -area + 5.0 * pen

def _smooth_min_area_symmetric(params, beta=50.0):
    """Smooth min area in symmetric parameter space."""
    pts = _params_to_points_symmetric(params)
    areas = _all_areas(pts)
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    logsumexp_val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -logsumexp_val / beta

def _objective_smooth_symmetric(params, beta=50.0):
    """Smooth objective in symmetric parameter space."""
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 0.1:
        return 1e6 + pen
    smin = _smooth_min_area_symmetric(params, beta)
    return -smin + 5.0 * pen

def _init_symmetric_params():
    """Create initial symmetric parameters."""
    params = np.zeros(11)
    # Center point at centroid height
    params[0] = H / 3.0
    
    # 5 pairs at different heights
    heights = [H * 0.15, H * 0.3, H * 0.5, H * 0.7, H * 0.85]
    for i, h in enumerate(heights):
        # x such that point is inside: x >= h/sqrt(3) and x <= 0.5
        x_min = h / np.sqrt(3)
        x = (x_min + 0.5) / 2.0  # midpoint of valid range
        params[1 + 2*i] = x
        params[2 + 2*i] = h
    
    return params

def _init_random_symmetric(rng):
    """Generate random symmetric parameters."""
    params = np.zeros(11)
    params[0] = rng.random() * H
    for i in range(5):
        y = rng.random() * H
        x_min = y / np.sqrt(3)
        x_max = 0.5
        if x_min >= x_max:
            x = 0.5
        else:
            x = x_min + rng.random() * (x_max - x_min)
        params[1 + 2*i] = x
        params[2 + 2*i] = y
    return params

def _refine_symmetric(params, deadline):
    """Refine symmetric parameters using Nelder-Mead on individual coordinates."""
    p = params.copy()
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(11):
            if time.time() > deadline - 5:
                break
            best_val = _objective_symmetric(p)
            
            def obj_i(param_i):
                p[i] = param_i
                return _objective_symmetric(p)
            
            try:
                res = minimize(obj_i, [p[i]], method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val - 1e-12:
                    p[i] = res.x[0]
                    improved = True
            except Exception:
                pass
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Bounds for symmetric parameters
    # params[0]: y of center point, in [0, H]
    # params[1+2i]: x of left point, in [0, 0.5]
    # params[2+2i]: y of left point, in [0, H]
    bounds_sym = [(0, H)]  # center y
    for i in range(5):
        bounds_sym.append((0, 0.5))  # x
        bounds_sym.append((0, H))    # y
    
    best_params_sym = None
    best_score = -np.inf
    
    def update_best_sym(params_sym):
        nonlocal best_params_sym, best_score
        if params_sym is not None:
            pts = _clamp_to_triangle(_params_to_points_symmetric(params_sym))
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = params_sym.copy()
    
    def update_best_full(points):
        nonlocal best_params_sym, best_score
        if points is not None:
            pts = _clamp_to_triangle(points)
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = _points_to_params_symmetric(pts)
    
    # Phase 1: Global optimization in symmetric space with dual_annealing
    if time.time() < deadline - 30:
        try:
            init_sym = _init_symmetric_params()
            res_da = dual_annealing(
                _objective_smooth_symmetric,
                bounds_sym,
                x0=init_sym,
                maxiter=150,
                seed=42,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                clip=True
            )
            update_best_sym(res_da.x)
        except Exception:
            pass
    
    # Phase 2: Global optimization with differential_evolution in symmetric space
    if time.time() < deadline - 25:
        try:
            res_de = differential_evolution(
                _objective_smooth_symmetric,
                bounds_sym,
                maxiter=80,
                popsize=40,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            update_best_sym(res_de.x)
        except Exception:
            pass
    
    # Phase 3: Local polish with exact objective in symmetric space
    if best_params_sym is not None and time.time() < deadline - 15:
        try:
            res_nm = minimize(
                _objective_symmetric,
                best_params_sym,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best_sym(res_nm.x)
        except Exception:
            pass
    
    # Phase 4: Coordinate-wise refinement in symmetric space
    if best_params_sym is not None and time.time() < deadline - 5:
        try:
            refined = _refine_symmetric(best_params_sym, deadline)
            update_best_sym(refined)
        except Exception:
            pass
    
    # Phase 5: Multi-start in symmetric space
    if time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 5:
                break
            init_params = _init_random_symmetric(rng)
            try:
                res_r = minimize(
                    _objective_symmetric,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                update_best_sym(res_r.x)
            except Exception:
                pass
    
    # Final: also try full-space local polish from best symmetric result
    if best_params_sym is not None and time.time() < deadline - 5:
        try:
            pts_full = _params_to_points_symmetric(best_params_sym)
            pts_clamped = _clamp_to_triangle(pts_full)
            
            def obj_full(params_full):
                pts = _clamp_to_triangle(params_full.reshape(N, 2))
                pen = _inside_penalty(pts)
                if pen > 0.1:
                    return 1e6 + pen
                return -_min_area(pts) + 5.0 * pen
            
            res_full = minimize(
                obj_full,
                pts_clamped.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best_full(res_full.x.reshape(N, 2))
        except Exception:
            pass
    
    # Final fallback
    if best_params_sym is None:
        best_params_sym = _init_symmetric_params()
    
    points = _clamp_to_triangle(_params_to_points_symmetric(best_params_sym))
    return points
