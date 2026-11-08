import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
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
    except Exception:
        return 1e-10

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=50.0):
    """
    Negative soft-min of normalized triangle areas using log-sum-exp.
    params: flattened array of shape (N*2,)
    """
    points = params.reshape(N, 2)
    
    # Center the points
    points = points - np.mean(points, axis=0)
    
    # Scale to unit max norm
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm < 1e-10:
        return 1e10
    points = points / max_norm
    
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    
    if hull_area <= 1e-10:
        return 1e10
    
    normalized = areas / hull_area
    
    # Soft minimum using log-sum-exp for numerical stability
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize soft_min, so minimize -soft_min
    max_val = np.max(normalized)
    log_sum = np.log(np.sum(np.exp(-beta * (normalized - max_val))))
    soft_min_val = max_val - np.log(np.sum(np.exp(-beta * (normalized - max_val)))) / beta
    
    return -soft_min_val

def _true_min_objective(params):
    """True minimum objective for final evaluation."""
    points = params.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm < 1e-10:
        return 1e10
    points = points / max_norm
    return -_score(points)

def _generate_regular_polygon_points():
    """Generate regular N-gon points."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1e10
    
    # Start with regular polygon as baseline
    regular = _generate_regular_polygon_points()
    reg_score = _score(regular)
    best_points = regular.copy()
    best_score = reg_score
    
    # Direct coordinate optimization with soft-min objective
    # Initialize from regular polygon
    init_params = regular.flatten()
    
    # Try multiple beta values and initializations
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() - t0 > deadline:
            break
        
        # Multiple random perturbations of the regular polygon
        for trial in range(3):
            if time.time() - t0 > deadline:
                break
            
            if trial == 0:
                start = init_params.copy()
            else:
                rng = np.random.default_rng(42 + trial)
                noise = rng.normal(0, 0.1, size=N*2)
                start = init_params + noise
            
            try:
                res = minimize(
                    _soft_min_objective, 
                    start, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.x is not None:
                    cand = res.x.reshape(N, 2)
                    cand = cand - np.mean(cand, axis=0)
                    max_norm = np.max(np.linalg.norm(cand, axis=1))
                    if max_norm > 1e-10:
                        cand = cand / max_norm
                        cand_score = _score(cand)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_points = cand.copy()
            except Exception:
                pass
    
    # Additional refinement with higher beta on best solution found
    if best_points is not None and time.time() - t0 < deadline - 10:
        for beta in [200.0, 500.0]:
            if time.time() - t0 > deadline:
                break
            
            start = best_points.flatten()
            try:
                res = minimize(
                    _soft_min_objective, 
                    start, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
                )
                
                if res.x is not None:
                    cand = res.x.reshape(N, 2)
                    cand = cand - np.mean(cand, axis=0)
                    max_norm = np.max(np.linalg.norm(cand, axis=1))
                    if max_norm > 1e-10:
                        cand = cand / max_norm
                        cand_score = _score(cand)
                        if cand_score > best_score:
                            best_score = cand_score
                            best_points = cand.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _generate_regular_polygon_points()
    
    return best_points
