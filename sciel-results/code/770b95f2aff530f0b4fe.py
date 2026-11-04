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

    # Vectorized triangle area computation
    # For each pair (i,j), compute cross products with all k > j
    areas = np.empty((N*(N-1)*(N-2))//6)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            v1 = points[j] - points[i]
            v2 = points[j+1:] - points[i]
            cross = v1[0]*v2[:,1] - v1[1]*v2[:,0]
            areas[idx:idx+len(cross)] = 0.5 * np.abs(cross)
            idx += len(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area


def _soft_min_score(points, beta=50.0):
    """Smooth approximation of min triangle area / hull area using log-sum-exp."""
    if points.shape != (N, 2):
        return -1e6
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return -1e6
    except Exception:
        return -1e6

    # Vectorized triangle area computation
    areas = np.empty((N*(N-1)*(N-2))//6)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            v1 = points[j] - points[i]
            v2 = points[j+1:] - points[i]
            cross = v1[0]*v2[:,1] - v1[1]*v2[:,0]
            areas[idx:idx+len(cross)] = 0.5 * np.abs(cross)
            idx += len(cross)
    
    # Soft min via log-sum-exp: soft_min(a) ≈ -log(sum(exp(-beta*a)))/beta
    # This is smooth and approximates min as beta -> inf
    scaled = -beta * areas / hull_area
    # Shift for numerical stability
    max_s = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - max_s))) + max_s
    soft_min_val = -log_sum / beta
    
    return soft_min_val

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
        # Use soft-min surrogate for smoother optimization
        score = _soft_min_score(pts, beta=30.0)
        return -score

    def objective_exact(params):
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
                # Phase 1: Coarse search with soft-min surrogate
                res = minimize(objective, x0, method='Nelder-Mead', options={'xatol':1e-5, 'fatol':1e-5, 'maxiter':150})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass
            if time.time() - start_time > time_limit * 0.6:
                break
        if time.time() - start_time > time_limit * 0.6:
            break

    # Phase 2: Fine-tune with exact objective if we have a candidate
    if best_params is not None and time.time() - start_time < time_limit * 0.8:
        try:
            res = minimize(objective_exact, best_params, method='Nelder-Mead', options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':300})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return 0, None
        
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
        if time.time() > deadline - 15:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0) # Just outer ring
            score = _calculate_score(pts)
            try_update(score, pts)
            continue
            
        # Dynamic time limit based on remaining time
        remaining = deadline - 15 - time.time()
        time_limit = min(12.0, max(3.0, remaining / 3.0))
        
        score, params = _optimize_two_ring(n_out, n_in, time_limit=time_limit)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            # Polish with full coordinate optimization using soft-min first
            def neg_soft_flat(x):
                p = x.reshape(N, 2)
                return -_soft_min_score(p, beta=30.0)
            
            def neg_exact_flat(x):
                p = x.reshape(N, 2)
                return -_calculate_score(p)
            
            try:
                # Phase 1: Soft-min polish
                res1 = minimize(neg_soft_flat, pts.flatten(), method='L-BFGS-B', options={'maxiter':80})
                pts1 = res1.x.reshape(N, 2)
                score1 = _calculate_score(pts1)
                
                # Phase 2: Exact polish from soft-min result
                res2 = minimize(neg_exact_flat, pts1.flatten(), method='Nelder-Mead', options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':200})
                pts2 = res2.x.reshape(N, 2)
                score2 = _calculate_score(pts2)
                
                new_score = max(score1, score2)
                new_pts = pts1 if score1 >= score2 else pts2
                if new_score > score:
                    score = new_score
                    pts = new_pts
            except Exception:
                pass
            
            try_update(score, pts)

    # Final polish on best points with tight tolerance
    if best_points is not None and time.time() < deadline - 5:
        def neg_exact_flat(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        
        try:
            res = minimize(neg_exact_flat, best_points.flatten(), method='Nelder-Mead', options={'xatol':1e-9, 'fatol':1e-9, 'maxiter':500})
            final_pts = res.x.reshape(N, 2)
            final_score = _calculate_score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
