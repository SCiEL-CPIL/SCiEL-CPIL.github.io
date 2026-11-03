"""
Optimization approach: Centrally-symmetric 14-point configuration.
7 pairs of antipodal points, each pair has its own angle and radius.
Parameters: 7 angles + 7 radii = 14 parameters.
Multi-restart with Nelder-Mead and Powell on the symmetric parameter space.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
NP = 7  # number of antipodal pairs

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _sym_params_to_points(params):
    """
    Convert symmetric parameters (angles, radii) to 14 points.
    params: array of shape (14,) = [a0, a1, ..., a6, r0, r1, ..., r6]
    where a_k are angles for the 7 pairs, r_k are radii.
    Point k is at (r_k * cos(a_k), r_k * sin(a_k))
    Point k+7 is at (-r_k * cos(a_k), -r_k * sin(a_k))
    """
    angles = params[:NP]
    radii = params[NP:]
    radii = np.maximum(radii, 0.01)
    
    x1 = radii * np.cos(angles)
    y1 = radii * np.sin(angles)
    x2 = -x1
    y2 = -y1
    
    pts = np.zeros((N, 2))
    pts[:NP, 0] = x1
    pts[:NP, 1] = y1
    pts[NP:, 0] = x2
    pts[NP:, 1] = y2
    return pts

def _neg_score_sym(params):
    """Negative score for minimization."""
    pts = _sym_params_to_points(params)
    return -_score(pts)

def _initial_regular_sym():
    """Regular 14-gon in symmetric parameterization."""
    angles = np.linspace(0, 2 * np.pi / NP, NP, endpoint=False)
    radii = np.ones(NP)
    return np.concatenate([angles, radii])

def _initial_jittered_sym(rng, base_angles=None, jitter_scale=0.1):
    """Generate a jittered symmetric initial guess."""
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi / NP, NP, endpoint=False)
    angles = base_angles + rng.uniform(-jitter_scale, jitter_scale, size=NP)
    radii = 1.0 + rng.uniform(-0.1, 0.1, size=NP)
    radii = np.maximum(radii, 0.1)
    return np.concatenate([angles, radii])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_params = None
    best_score_val = -np.inf
    
    def _try_optimize(params_init, method='Nelder-Mead', maxiter=2000):
        nonlocal best_params, best_score_val
        if time.time() > deadline - 10:
            return
        
        try:
            if method == 'Nelder-Mead':
                res = minimize(_neg_score_sym, params_init, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(_neg_score_sym, params_init, method='Powell',
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-12})
            else:
                return
            
            if res.success or res.fun < 0:
                candidate_params = res.x
                candidate_score = -res.fun
                if candidate_score > best_score_val:
                    best_score_val = candidate_score
                    best_params = candidate_params.copy()
        except Exception:
            pass
    
    # Phase 1: Optimize from regular polygon
    init_reg = _initial_regular_sym()
    _try_optimize(init_reg, method='Nelder-Mead', maxiter=3000)
    
    # Phase 2: Powell refinement
    if best_params is not None:
        _try_optimize(best_params, method='Powell', maxiter=2000)
    
    # Phase 3: Multiple restarts with different jittered initial guesses
    rng = np.random.default_rng(42)
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        jitter_scale = 0.05 + 0.03 * i
        init_jitter = _initial_jittered_sym(rng, jitter_scale=jitter_scale)
        
        # First Nelder-Mead
        _try_optimize(init_jitter, method='Nelder-Mead', maxiter=1500)
        
        # Then Powell if we have time
        if best_params is not None and time.time() < deadline - 15:
            _try_optimize(best_params, method='Powell', maxiter=1000)
    
    # Phase 4: Final Nelder-Mead polish on best
    if best_params is not None and time.time() < deadline - 5:
        _try_optimize(best_params, method='Nelder-Mead', maxiter=2000)
    
    # Convert best params to points
    if best_params is None:
        best_params = _initial_regular_sym()
    
    pts = _sym_params_to_points(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            pts = pts * scale
    except Exception:
        pass
    
    return pts
