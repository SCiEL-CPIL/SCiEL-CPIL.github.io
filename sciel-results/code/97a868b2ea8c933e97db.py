"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _all_tri_indices():
    """Precompute indices for all triangles."""
    indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                indices.append((i, j, k))
    return np.array(indices)

_TRI_INDICES = _all_tri_indices()

def _compute_all_areas(pts):
    """Compute all triangle areas for a set of points."""
    pts = np.asarray(pts)
    areas = np.zeros(len(_TRI_INDICES))
    for idx, (i, j, k) in enumerate(_TRI_INDICES):
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        areas[idx] = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                p2[0] * (p3[1] - p1[1]) + 
                                p3[0] * (p1[1] - p2[1]))
    return areas

def _soft_min_triangle_ratio(pts, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area ratio.
    Uses log-sum-exp trick for soft-min.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    areas = _compute_all_areas(pts)
    
    # Soft-min using log-sum-exp
    # min(areas) ≈ -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max of (-beta * areas)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _objective_flat(params, beta=50.0):
    """
    Objective function for minimizer.
    Minimizes -soft_min_triangle_ratio.
    """
    pts = params.reshape(N, 2)
    val = _soft_min_triangle_ratio(pts, beta=beta)
    return -val

def _generate_diverse_starts(rng, n_starts=20):
    """Generate diverse starting configurations."""
    starts = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    p0 = np.column_stack((x0, y0)).flatten()
    starts.append(p0)
    
    # 2. Halton sequence
    def _halton_sequence(n, base):
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0
            i_val = i + 1
            while i_val > 0:
                f /= base
                i_val, rem = divmod(i_val, base)
                seq[i] += rem * f
        return seq
    
    x = _halton_sequence(N, 2) - 0.5
    y = _halton_sequence(N, 3) - 0.5
    pts = np.column_stack((x, y))
    norms = np.linalg.norm(pts, axis=1)
    if np.max(norms) > 0:
        pts = pts / np.max(norms)
    starts.append(pts.flatten())
    
    # 3. Random perturbations of regular polygon
    for _ in range(n_starts - 2):
        noise = rng.normal(0, 0.3, size=N*2)
        p_pert = p0 + noise
        p_pert = p_pert - np.mean(p_pert, axis=0)
        starts.append(p_pert)
    
    return starts

def _optimize_from_start(p0, start_time, time_limit, beta_schedule=None):
    """
    Run optimization from a single starting point with a schedule of beta values.
    Returns best points found.
    """
    best_score = _compute_min_tri_area_norm(p0.reshape(N, 2))
    best_pts = p0.copy()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 30.0, 60.0, 120.0, 200.0]
    
    current_pts = p0.copy()
    
    for beta in beta_schedule:
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        def objective(params):
            return _objective_flat(params, beta=beta)
        
        try:
            res = minimize(
                objective, 
                current_pts, 
                method='L-BFGS-B', 
                options={'maxiter': 3000, 'maxfun': 80000, 'disp': False}
            )
            
            if res.success:
                score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.copy()
                current_pts = res.x.copy()
                
        except Exception:
            pass
            
    return best_pts, best_score

def _local_search_around(pts, start_time, time_limit, n_perturb=10, sigma=0.1):
    """
    Perform local search by perturbing the current best solution.
    """
    best_pts = pts.copy()
    best_score = _compute_min_tri_area_norm(pts)
    
    rng = np.random.default_rng(123)
    
    for i in range(n_perturb):
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        noise = rng.normal(0, sigma, size=pts.shape)
        perturbed = pts + noise
        perturbed = perturbed - np.mean(perturbed, axis=0)
        
        score = _compute_min_tri_area_norm(perturbed)
        if score > best_score:
            best_score = score
            best_pts = perturbed.copy()
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = _generate_diverse_starts(rng, n_starts=15)
    
    best_pts = None
    best_score = -1e9
    
    # Phase 1: Try each start point with gradient optimization
    for i, start in enumerate(starts):
        if (time.time() - start_time) > time_limit - 15.0:
            break
            
        pts, score = _optimize_from_start(start, start_time, time_limit)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 2: Local search around the best solution
    if best_pts is not None and (time.time() - start_time) < time_limit - 10.0:
        best_pts, best_score = _local_search_around(best_pts, start_time, time_limit, 
                                                     n_perturb=15, sigma=0.05)
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and (time.time() - start_time) < time_limit - 5.0:
        def objective_polish(params):
            return _objective_flat(params, beta=300.0)
        
        try:
            res = minimize(
                objective_polish, 
                best_pts, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'maxfun': 100000, 'disp': False}
            )
            
            if res.success:
                score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.copy()
                    
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x0 = np.cos(angles)
        y0 = np.sin(angles)
        return np.column_stack((x0, y0))
        
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
