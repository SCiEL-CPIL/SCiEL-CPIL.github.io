import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns a smooth approximation of the minimum area.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp
    max_a = np.max(areas)
    # Numerical stability: shift by max
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _neg_soft_score(points_flat, beta=100.0):
    """
    Negative soft-score for gradient-based optimization.
    This is the objective function for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e10 # Large penalty for degenerate hull
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 1e10

def _neg_hard_score(points_flat):
    """
    Negative hard score for non-gradient methods like Nelder-Mead.
    """
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1. The score is invariant to scaling, so this is safe."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        # Scale by 1/sqrt(ha) so that new hull area = ha / ha = 1
        # Triangle areas scale by 1/ha, so min_triangle_area_new = min_triangle_area / ha
        # Score_new = (min_triangle_area / ha) / 1 = min_triangle_area / ha = Score_old
        # So the score is preserved!
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_points(r_inner, delta_inner, jitter_o, jitter_i, n_outer=7, n_inner=7):
    """
    Generate points with flexible angular positions on two rings.
    """
    base_angles_o = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    base_angles_i = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    
    outer_angles = base_angles_o + jitter_o
    inner_angles = base_angles_i + delta_inner + jitter_i
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

def _generate_three_ring_points(r1, r2, delta1, delta2, jitter, n1, n2, n3):
    """
    Generate points on three concentric rings.
    """
    base1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    base2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)
    base3 = np.linspace(0, 2 * np.pi, n3, endpoint=False)
    
    a1 = base1 + jitter[:n1]
    a2 = base2 + delta1 + jitter[n1:n1+n2]
    a3 = base3 + delta2 + jitter[n1+n2:]
    
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = np.column_stack([r1 * np.cos(a2), r1 * np.sin(a2)])
    p3 = np.column_stack([r2 * np.cos(a3), r2 * np.sin(a3)])
    
    return np.vstack([p1, p2, p3])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # 1. Generate diverse initial points focused on 2-ring and 3-ring structures
    initial_points = []
    rng = np.random.RandomState(42)
    
    # Regular 14-gon (baseline)
    initial_points.append(_regular_polygon(14, 1.0, 0.0))
    initial_points.append(_regular_polygon(14, 1.0, np.pi/14.0))
    
    # Two-ring structures: 7 outer + 7 inner
    # These are the most promising based on inspiration
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for delta in [0.0, np.pi/7.0, np.pi/14.0]:
            jitter_o = np.zeros(7)
            jitter_i = np.zeros(7)
            pts = _generate_two_ring_points(r_inner, delta, jitter_o, jitter_i)
            initial_points.append(pts)
            
            # Add jittered versions
            for _ in range(3):
                jitter_o = rng.normal(0, 0.15, 7)
                jitter_i = rng.normal(0, 0.15, 7)
                pts_j = _generate_two_ring_points(r_inner, delta, jitter_o, jitter_i)
                initial_points.append(pts_j)
    
    # Three-ring structures: 5+5+4 or 6+4+4
    for r1, r2 in [(0.4, 0.7), (0.5, 0.8), (0.6, 0.85)]:
        for delta1 in [0.0, np.pi/5.0]:
            for delta2 in [0.0, np.pi/4.0]:
                jitter = np.zeros(14)
                pts = _generate_three_ring_points(r1, r2, delta1, delta2, jitter, 5, 5, 4)
                initial_points.append(pts)
                
                jitter = rng.normal(0, 0.1, 14)
                pts_j = _generate_three_ring_points(r1, r2, delta1, delta2, jitter, 5, 5, 4)
                initial_points.append(pts_j)
    
    # 2. Optimize each initial point
    for init_pts in initial_points:
        if _time_left() < 3:
            break
            
        init_flat = init_pts.flatten()
        
        # Step A: Coarse local optimization with Nelder-Mead on hard score
        try:
            res_nm = scipy.optimize.minimize(
                _neg_hard_score,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            current_flat = res_nm.x
        except Exception:
            current_flat = init_flat

        # Step B: Progressive refinement with L-BFGS-B on soft score
        betas = [5.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if _time_left() < 1.5:
                break
                
            try:
                res_lbfgs = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, beta),
                    current_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_flat = res_lbfgs.x
                
                pts_check = _normalize(current_flat.reshape(-1, 2))
                sc_check = _score(pts_check)
                _update_best(pts_check, sc_check)
                
            except Exception:
                pass

    # 3. Final high-precision polish on the best point
    if best_points is not None and _time_left() > 5:
        # Try multiple polish passes with increasing beta
        for beta in [100.0, 200.0, 500.0, 1000.0]:
            if _time_left() < 1:
                break
            try:
                res_polish = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, beta),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
            except Exception:
                pass
        
        # Random restarts from best point
        for _ in range(10):
            if _time_left() < 2:
                break
            noise = rng.normal(0, 0.03, 28)
            perturbed = best_points.flatten() + noise
            try:
                res_polish = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, 200.0),
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-13}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
