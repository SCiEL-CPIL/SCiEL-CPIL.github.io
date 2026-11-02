import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14
N_RING = 7  # 7 points per ring, 2 rings

def _compute_min_triangle_area_vec(points):
    """
    Vectorized computation of minimum triangle area.
    points: (N, 2) array
    Returns: float
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all cross products using broadcasting
    # For each pair (i,j), compute cross with all k > j
    # Area = 0.5 * |(p_j - p_i) x (p_k - p_i)|
    
    # Use a more memory-efficient approach for N=14
    # 364 triangles total, vectorize over k for each (i,j)
    min_area = float('inf')
    
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            d_ij = pj - pi
            # Vectorized over k
            d_ik = points[j+1:] - pi
            # Cross products: d_ij[0]*d_ik[1] - d_ij[1]*d_ik[0]
            crosses = d_ij[0] * d_ik[:, 1] - d_ij[1] * d_ik[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-15:
                    return 0.0
    
    return min_area

def _compute_hull_area_fast(points):
    """
    Compute convex hull area.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    
    hull_area = _compute_hull_area_fast(points)
    if hull_area < 1e-12:
        return 0.0
        
    min_area = _compute_min_triangle_area_vec(points)
    if min_area <= 0.0:
        return 0.0
        
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 2-ring parameters to N=14 points.
    params: array of length 2 + 2*N_RING*2 = 2 + 28 = 30
    [0]: log(r_inner)
    [1]: log(r_outer)
    [2:2+7]: angle offsets for inner ring (relative to base angle)
    [2+7:2+14]: angle offsets for outer ring
    
    Base angles for each ring are evenly spaced: 2*pi*k/7 for k=0..6
    The angle offsets allow small perturbations.
    """
    log_r_in = params[0]
    log_r_out = params[1]
    theta_in = params[2:2+N_RING]
    theta_out = params[2+N_RING:2+2*N_RING]
    
    r_in = np.exp(log_r_in)
    r_out = np.exp(log_r_out)
    
    base_angles_in = 2 * np.pi * np.arange(N_RING) / N_RING
    base_angles_out = 2 * np.pi * np.arange(N_RING) / N_RING + np.pi / N_RING  # offset by half-step
    
    angles_in = base_angles_in + theta_in
    angles_out = base_angles_out + theta_out
    
    x_in = r_in * np.cos(angles_in)
    y_in = r_in * np.sin(angles_in)
    x_out = r_out * np.cos(angles_out)
    y_out = r_out * np.sin(angles_out)
    
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    
    return np.column_stack((x, y))

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    points = _params_to_points(params)
    score = _score(points)
    if score <= 1e-12:
        return 1e10
    return -score

def _objective_soft(params, beta=50.0):
    """
    Soft objective using log-sum-exp for smoother gradients.
    Instead of min area, use a soft-min that approximates min but is differentiable.
    """
    points = _params_to_points(params)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 1e10
    
    hull_area = _compute_hull_area_fast(points)
    if hull_area < 1e-12:
        return 1e10
    
    # Compute all triangle areas
    n = len(points)
    areas = np.empty(364)
    idx = 0
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            d_ij = pj - pi
            d_ik = points[j+1:] - pi
            crosses = d_ij[0] * d_ik[:, 1] - d_ij[1] * d_ik[:, 0]
            a = 0.5 * np.abs(crosses)
            areas[idx:idx+len(a)] = a
            idx += len(a)
    
    if np.any(areas <= 0):
        return 1e10
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This approximates min(areas) as beta -> inf
    log_sum_exp = np.log(np.sum(np.exp(-beta * areas)))
    soft_min_area = -log_sum_exp / beta
    
    return -(soft_min_area / hull_area)

def _generate_initial_params():
    """
    Generate diverse initial parameter sets for 2-ring parameterization.
    Returns a list of param arrays of length 30.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 2-ring: r_in=0.5, r_out=1.0, zero angle offsets
    for r_in in [0.3, 0.5, 0.6, 0.7, 0.8]:
        for r_out in [0.9, 1.0, 1.1, 1.2]:
            if r_in < r_out:
                params = np.zeros(2 + 2*N_RING)
                params[0] = np.log(r_in)
                params[1] = np.log(r_out)
                inits.append(params)
    
    # 2. Random perturbations
    for _ in range(30):
        r_in = rng.uniform(0.3, 0.9)
        r_out = rng.uniform(0.9, 1.5)
        if r_in >= r_out:
            r_in, r_out = r_out, r_in
        params = np.zeros(2 + 2*N_RING)
        params[0] = np.log(r_in)
        params[1] = np.log(r_out)
        params[2:2+N_RING] = rng.normal(0, 0.1, N_RING)
        params[2+N_RING:2+2*N_RING] = rng.normal(0, 0.1, N_RING)
        inits.append(params)
    
    # 3. Larger random perturbations
    for _ in range(20):
        r_in = rng.uniform(0.2, 1.0)
        r_out = rng.uniform(0.8, 2.0)
        if r_in >= r_out:
            r_in, r_out = r_out, r_in
        params = np.zeros(2 + 2*N_RING)
        params[0] = np.log(r_in)
        params[1] = np.log(r_out)
        params[2:2+N_RING] = rng.normal(0, 0.3, N_RING)
        params[2+N_RING:2+2*N_RING] = rng.normal(0, 0.3, N_RING)
        inits.append(params)
    
    return inits

def _bounds_2ring():
    """
    Bounds for 2-ring parameters.
    """
    bounds = []
    # log(r_in): r in [0.1, 1.5]
    bounds.append((np.log(0.1), np.log(1.5)))
    # log(r_out): r in [0.5, 3.0]
    bounds.append((np.log(0.5), np.log(3.0)))
    # Angle offsets: [-0.5, 0.5] radians
    for _ in range(2 * N_RING):
        bounds.append((-0.5, 0.5))
    return bounds

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    inits = _generate_initial_params()
    bounds = _bounds_2ring()
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.3, 1.0),
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
    
    # Phase 2: Local refinement from multiple starts
    for init_params in inits:
        if time.time() > deadline - 30:
            break
        
        # Nelder-Mead
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if np.isfinite(res.fun):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = _params_to_points(res.x)
                    
        except Exception:
            continue
        
        # Powell refinement
        if best_score > 0 and time.time() < deadline - 15:
            try:
                res2 = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 1000,
                        'xtol': 1e-12,
                        'ftol': 1e-14
                    }
                )
                score2 = -res2.fun
                if np.isfinite(score2) and score2 > best_score:
                    best_score = score2
                    best_params = res2.x.copy()
                    best_points = _params_to_points(res2.x)
            except Exception:
                pass
        
        # L-BFGS-B with soft objective for smoother landscape
        if best_score > 0 and time.time() < deadline - 10:
            try:
                res3 = minimize(
                    _objective_soft,
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                if np.isfinite(res3.fun):
                    # Evaluate with hard objective
                    score3 = -_objective(res3.x)
                    if score3 > best_score:
                        best_score = score3
                        best_params = res3.x.copy()
                        best_points = _params_to_points(res3.x)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 2000,
                    'xtol': 1e-14,
                    'ftol': 1e-16
                }
            )
            if np.isfinite(res_final.fun):
                score_final = -res_final.fun
                if score_final > best_score:
                    best_score = score_final
                    best_params = res_final.x.copy()
                    best_points = _params_to_points(res_final.x)
        except Exception:
            pass
    
    # Final fallback if nothing found
    if best_points is None:
        # Return regular 2-ring
        params_reg = np.zeros(2 + 2*N_RING)
        params_reg[0] = np.log(0.5)
        params_reg[1] = np.log(1.0)
        best_points = _params_to_points(params_reg)
        
    # Normalize to unit area
    hull_area = _compute_hull_area_fast(best_points)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale
    else:
        best_points = best_points * 1.0
        
    return best_points
