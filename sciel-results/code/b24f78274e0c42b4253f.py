import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area.
    We minimize - (soft_min(areas) / hull_area).
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def _two_shell_points(r_outer, r_inner, phase, n_outer=7, n_inner=6):
    """Create two-shell configuration: n_outer on outer circle, n_inner on inner circle."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(phase, 2 * np.pi + phase, n_inner, endpoint=False)
    outer_pts = np.column_stack((r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)))
    inner_pts = np.column_stack((r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)))
    return np.vstack([outer_pts, inner_pts])

def _two_shell_objective(params, beta=50.0):
    """Objective for optimizing two-shell parameters: [r_outer, r_inner, phase]."""
    r_outer, r_inner, phase = params
    if r_outer <= 0 or r_inner < 0:
        return 1e10
    points = _two_shell_points(r_outer, r_inner, phase)
    return _soft_min_objective(points.flatten(), beta=beta)

def construct():
    n = N
    t0 = time.time()
    deadline = 130.0
    
    # Phase 1: Optimize two-shell parameters (3D problem - fast)
    best_points = None
    best_score = 0.0
    
    # Try multiple initial guesses for the two-shell optimization
    init_guesses = [
        [1.0, 0.5, 0.0],
        [1.0, 0.6, np.pi/6],
        [1.0, 0.4, np.pi/6],
        [1.0, 0.7, np.pi/7],
        [1.0, 0.5, np.pi/7],
        [1.0, 0.55, 0.0],
        [1.0, 0.45, np.pi/6],
        [1.0, 0.65, np.pi/7],
    ]
    
    for guess in init_guesses:
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize(_two_shell_objective, guess, method='L-BFGS-B',
                          bounds=[(0.1, 1.5), (0.0, 1.0), (-np.pi, np.pi)],
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() - t0 > deadline:
                break
                
            cand_params = np.clip(res.x, [0.1, 0.0, -np.pi], [1.5, 1.0, np.pi])
            candidate = _two_shell_points(cand_params[0], cand_params[1], cand_params[2])
            cand_score = _score(candidate)
            
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # If two-shell didn't give anything, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    # Phase 2: Full-coordinate polish from best two-shell config
    if time.time() - t0 < deadline:
        try:
            x0 = best_points.flatten()
            res1 = minimize(_soft_min_objective, x0, args=(50.0,), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res1.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Powell method polish
    if time.time() - t0 < deadline:
        try:
            x0 = best_points.flatten()
            res2 = minimize(_soft_min_objective, x0, args=(100.0,), method='Powell',
                           options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res2.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    # Phase 4: Nelder-Mead high-precision polish
    if time.time() - t0 < deadline:
        try:
            x0 = best_points.flatten()
            res3 = minimize(_soft_min_objective, x0, args=(200.0,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res3.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    return best_points
