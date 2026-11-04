import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area computation
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    # Soft minimum via log-sum-exp
    max_area = np.max(areas)
    if max_area < 1e-12:
        return 0.0
    shifted = areas - max_area
    log_sum = max_area + np.log(np.sum(np.exp(shifted / temperature)))
    soft_min = max_area - temperature * np.log(np.sum(np.exp(shifted / temperature)))

    return soft_min / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    for r0 in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]:
        for phi0 in [0, np.pi/n_in if n_in > 0 else 0, np.pi/(n_out if n_out > 0 else 1)]:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    return best_score, best_params

def _polish_points(points, time_limit=5.0):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    start_time = time.time()
    
    def neg_score_flat(x):
        p = x.reshape(N, 2)
        return -_calculate_score(p)
    
    try:
        res = minimize(neg_score_flat, points.flatten(), method='L-BFGS-B',
                      options={'maxiter':200, 'ftol':1e-12, 'gtol':1e-10})
        new_pts = res.x.reshape(N, 2)
        new_score = _calculate_score(new_pts)
        
        if new_score > 0:
            # Second polish pass with even tighter tolerance
            if time.time() - start_time < time_limit * 0.7:
                res2 = minimize(neg_score_flat, new_pts.flatten(), method='L-BFGS-B',
                              options={'maxiter':100, 'ftol':1e-14, 'gtol':1e-12})
                pts2 = res2.x.reshape(N, 2)
                score2 = _calculate_score(pts2)
                if score2 > new_score:
                    return pts2, score2
            return new_pts, new_score
    except Exception:
        pass
    
    return points, _calculate_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    splits = [
        (14, 0),
        (13, 1),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (8, 6),
        (7, 7)
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 15:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0)
            score = _calculate_score(pts)
            # Polish the regular polygon
            polished_pts, polished_score = _polish_points(pts, time_limit=3.0)
            try_update(polished_score, polished_pts)
            continue
        
        score, params = _optimize_two_ring(n_out, n_in, time_limit=8.0)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            # Polish with full coordinate optimization
            polished_pts, polished_score = _polish_points(pts, time_limit=5.0)
            try_update(polished_score, polished_pts)

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        # Try to polish even the fallback
        polished_pts, polished_score = _polish_points(best_points, time_limit=5.0)
        best_points = polished_pts
        best_score = polished_score

    return best_points
