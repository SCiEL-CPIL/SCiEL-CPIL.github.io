"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """
    Highly optimized minimum triangle area calculation.
    Uses vectorized operations for each pair (i, j) to compute areas with all k > j.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # We only need to check i < j < k
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        # Vectorized differences from i to j+1..n-1
        x_ji = x[i+1:] - xi
        y_ji = y[i+1:] - yi
        
        # For each j in i+1..n-2, we need cross products with k > j
        # This is slightly tricky to fully vectorize across j and k simultaneously 
        # without a large temporary array, but for N=14 it's fine.
        # Let's iterate j, but vectorize k.
        for j in range(i + 1, n - 1):
            # Vector for all k > j
            # x_k = x[j+1:], y_k = y[j+1:]
            # v1 = (x_j - x_i, y_j - y_i)
            # v2_k = (x_k - x_i, y_k - y_i)
            # cross = v1_x * v2_y - v1_y * v2_x
            
            v1_x = x[j] - xi
            v1_y = y[j] - yi
            
            v2_x = x[j+1:] - xi
            v2_y = y[j+1:] - yi
            
            cross = v1_x * v2_y - v1_y * v2_x
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                # Early exit if area is effectively zero
                if min_area < 1e-14:
                    return min_area
    return min_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        x_ji = x[i+1:] - xi
        y_ji = y[i+1:] - yi
        for j in range(i + 1, n - 1):
            v1_x = x[j] - xi
            v1_y = y[j] - yi
            v2_x = x[j+1:] - xi
            v2_y = y[j+1:] - yi
            cross = v1_x * v2_y - v1_y * v2_x
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

# ---------- Centrally symmetric parameterization ----------
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))
# This ensures central symmetry by construction.

def _params_to_points_flat(params):
    """Decode 14-parameter vector [angles(7), radii(7)] to 14 points."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters: minimize hull_area / min_triangle_area."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_flat(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_sym_soft(params, beta=200.0):
    """Soft objective for centrally symmetric parameters using log-sum-exp for soft-min."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_flat(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, shift by max
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
        
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    # Regular heptagon-based
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
        
    # Random perturbations
    rng = np.random.default_rng(42)
    for seed in range(20):
        rng = np.random.default_rng(seed * 11 + 5)
        angles = np.sort(rng.uniform(0, 2*np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.2, 7)
        radii = np.maximum(radii, 0.5)
        inits.append(np.concatenate([angles, radii]))
        
    return inits

# ---------- Optimizers ----------

def _nm(x0, deadline, max_iter=2000, obj=_objective_sym):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_sym_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res
    except Exception:
        return None

def _powell(x0, deadline, max_iter=2000, obj=_objective_sym):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
        return res
    except Exception:
        return None

def _de_sym(bounds, deadline, max_iter=200, popsize=20):
    """Differential evolution on symmetric parameters."""
    remaining = deadline - time.time()
    if remaining < 15.0:
        return None
    try:
        res = differential_evolution(_objective_sym, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-10, atol=1e-12,
                                      seed=42, polish=True, init='latinhypercube')
        return res
    except Exception:
        return None

# ---------- Main ----------

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, keep 10s buffer for safety, so 110s for search.
    # However, the prompt says "finish within 120 s". Let's use 115s to be safe.
    deadline = start_time + 115.0

    best_score = 0.0
    best_pts = None
    best_params = None

    def _consider_from_params(params):
        nonlocal best_score, best_pts, best_params
        points = _params_to_points_flat(params)
        if points.shape != (N, 2) or np.any(~np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_pts = points.copy()
            best_params = params.copy()

    def _consider_points(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Global Search with Differential Evolution
    bounds = [(-np.pi, np.pi)] * 7 + [(0.3, 2.0)] * 7
    res = _de_sym(bounds, deadline, max_iter=150, popsize=20)
    if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
        _consider_from_params(res.x)

    # Phase 2: Local Refinement from diverse starts
    inits = _generate_sym_inits()
    # Add the best from DE to inits if it exists
    if best_params is not None:
        inits.insert(0, best_params)

    for x0 in inits:
        if time.time() > deadline - 40:
            break
        # Nelder-Mead is robust for non-smooth hard objective
        res = _nm(x0, deadline, max_iter=1500, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 3: Gradient-based polish with soft objective
    # Use the best hard objective point to start soft optimization
    if best_params is not None and time.time() < deadline - 25:
        # Try L-BFGS on soft objective
        res = _lbfgs(best_params, deadline, max_iter=3000, obj=_objective_sym_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)
            
        # If L-BFGS improved, try again with even tighter tolerance or different beta
        # For now, just one pass is usually enough if the start is good.

    # Phase 4: Final high-precision polish with Powell (derivative-free, good for final steps)
    if best_params is not None and time.time() < deadline - 10:
        res = _powell(best_params, deadline, max_iter=2000, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 5: Final NM polish with very tight tolerances
    if best_params is not None and time.time() < deadline - 5:
        res = _nm(best_params, deadline, max_iter=1000, obj=_objective_sym)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Fallback: regular 14-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
