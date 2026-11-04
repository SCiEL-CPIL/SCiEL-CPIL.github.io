import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

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

    # Compute all triangle areas
    # Using vectorized cross product
    # i, j, k
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    
    min_area = np.min(areas)
    return min_area / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    # Outer ring: n_out points at radius 1.0
    # Inner ring: n_in points at radius r_in
    # Angles interleaved or aligned?
    # Let's assume aligned with offset phi_rel
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    
    return np.vstack((pts_out, pts_in))

def _optimize_two_ring(n_out, n_in, time_limit=10.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    # Try multiple initial guesses
    best_score = 0
    best_params = None
    
    for r0 in [0.2, 0.5, 0.8, 1.0]:
        for phi0 in [0, np.pi/n_in, np.pi/n_out]:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'xatol':1e-6, 'fatol':1e-6, 'maxiter':200})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    if best_params is None:
        return 0, None
        
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Test various splits
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
        if time.time() > deadline - 10:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0) # Just outer ring
            score = _calculate_score(pts)
            try_update(score, pts)
            continue
            
        score, params = _optimize_two_ring(n_out, n_in, time_limit=10.0)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            # Polish with full coordinate optimization
            def neg_score_flat(x):
                p = x.reshape(N, 2)
                return -_calculate_score(p)
            
            try:
                res = minimize(neg_score_flat, pts.flatten(), method='L-BFGS-B', options={'maxiter':100})
                new_pts = res.x.reshape(N, 2)
                new_score = _calculate_score(new_pts)
                if new_score > score:
                    score = new_score
                    pts = new_pts
            except Exception:
                pass
            
            try_update(score, pts)

    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
