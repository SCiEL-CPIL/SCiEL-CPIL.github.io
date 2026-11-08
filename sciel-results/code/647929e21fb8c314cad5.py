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

def _soft_min_objective(params, beta=200.0):
    """
    Negative soft-min triangle area / hull area.
    params: flattened array of shape (N*2,)
    Uses log-sum-exp for numerical stability.
    """
    points = params.reshape(N, 2)
    # Center points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    areas = _all_triangle_areas(points)
    if np.any(areas < 0):
        return 1e10
    
    # Soft min via log-sum-exp
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    if max_val < -700:  # all exp terms underflow
        lse = max_val
    else:
        lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_val = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    
    return -soft_min_val / hull_area

def _true_min_objective(params):
    """Negative true min triangle area / hull area (for final polish)."""
    points = params.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    
    return -np.min(areas) / hull_area

def _generate_regular_polygon_points():
    """Generate regular N-gon points centered at origin."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _generate_starts(rng, n_starts=30):
    """Generate diverse starting configurations."""
    starts = []
    
    # 1. Regular polygon
    regular = _generate_regular_polygon_points()
    starts.append(regular.flatten())
    
    # 2. Regular polygon with small random perturbations
    for _ in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        perturbed = regular + noise
        starts.append(perturbed.flatten())
    
    # 3. Two-ring configurations (free coordinates, not constrained to exact circles)
    for _ in range(8):
        n_inner = rng.integers(3, 8)
        n_outer = N - n_inner
        r_inner = rng.uniform(0.3, 0.7)
        r_outer = rng.uniform(0.9, 1.3)
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi/n_inner)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi/n_outer)
        pts = []
        for a in angles_inner:
            pts.append([r_inner * np.cos(a), r_inner * np.sin(a)])
        for a in angles_outer:
            pts.append([r_outer * np.cos(a), r_outer * np.sin(a)])
        pts = np.array(pts)
        # Add small noise to break exact symmetry
        pts += rng.normal(0, 0.02, (N, 2))
        starts.append(pts.flatten())
    
    # 4. Random points in disk
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(pts.flatten())
    
    # 5. Grid-like perturbation
    for _ in range(3):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + rng.normal(0, 0.1, N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(pts.flatten())
    
    return starts

def construct():
    t0 = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1e10
    
    # Start with regular polygon as baseline
    regular = _generate_regular_polygon_points()
    reg_score = _score(regular)
    best_points = regular.copy()
    best_score = reg_score
    
    # Generate diverse starts
    starts = _generate_starts(rng, n_starts=30)
    
    # Phase 1: L-BFGS-B with soft-min objective from each start
    for i, start in enumerate(starts):
        if time.time() - t0 > deadline:
            break
        
        try:
            # L-BFGS-B with soft-min (smooth, gradient-friendly)
            res = minimize(_soft_min_objective, start, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                cand = cand - np.mean(cand, axis=0)
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish with true min objective on best candidates
    if time.time() - t0 < deadline - 10.0:
        # Refine the best solution found
        try:
            res = minimize(_true_min_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                cand = cand - np.mean(cand, axis=0)
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 3: If time remains, do additional L-BFGS-B from perturbed best
    if time.time() - t0 < deadline - 5.0:
        for _ in range(5):
            if time.time() - t0 > deadline:
                break
            try:
                perturbed = best_points + rng.normal(0, 0.03, (N, 2))
                res = minimize(_soft_min_objective, perturbed.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 3000})
                if res.x is not None:
                    cand = res.x.reshape(N, 2)
                    cand = cand - np.mean(cand, axis=0)
                    cand_score = _score(cand)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = cand.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _generate_regular_polygon_points()
    
    return best_points
