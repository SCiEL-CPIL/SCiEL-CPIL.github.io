import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(points):
    # points: 8x2 array
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 0)
    D = np.max(dists)
    
    if D == 0:
        return 0.0
        
    scaled_points = points * (2.0 / D)
    
    diff_s = scaled_points[:, np.newaxis, :] - scaled_points[np.newaxis, :, :]
    dists_s = np.sqrt(np.sum(diff_s**2, axis=-1))
    np.fill_diagonal(dists_s, 0)
    
    # Avoid log(0)
    dists_s[dists_s < 1e-9] = 1e-9
    
    log_prod = np.sum(np.log(dists_s))
    return -log_prod + 8 * np.log(8)

def _get_points(r1, r2, delta, cx):
    # Square 1 at origin, radius r1
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2 at (cx, 0), radius r2, rotated by delta
    angles2 = angles1 + delta
    p2 = np.column_stack([r2 * np.cos(angles2) + cx, r2 * np.sin(angles2)])
    
    return np.vstack([p1, p2])

def _compute_score_wrapper(params):
    try:
        r1, r2, delta, cx = params
        if r1 < 0.01 or r2 < 0.01:
            return 1e10
        points = _get_points(r1, r2, delta, cx)
        return _compute_score(points)
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    best_score = np.inf
    best_params = None
    
    # Grid search over r1, r2, delta, cx
    # We can fix r1=1 to reduce dimensionality? 
    # Let's search r1, r2, delta, cx but with coarse grid.
    # Or fix r1=1 and search r2, delta, cx.
    # Let's fix r1=1.0 to make it 3D, which is faster and sufficient for relative geometry.
    
    r1_fixed = 1.0
    
    # Grid points
    r2_vals = np.linspace(0.2, 1.5, 10)
    delta_vals = np.linspace(0.0, np.pi/2, 10)
    cx_vals = np.linspace(-0.5, 0.5, 10)
    
    candidates = []
    
    for r2 in r2_vals:
        for delta in delta_vals:
            for cx in cx_vals:
                if time.time() > deadline:
                    break
                params = (r1_fixed, r2, delta, cx)
                score = _compute_score_wrapper(params)
                candidates.append((score, params))
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not candidates:
        # Fallback
        return [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]]

    # Sort by score (ascending, since we minimize negative log score)
    candidates.sort(key=lambda x: x[0])
    
    # Refine top 5 candidates
    for i in range(min(5, len(candidates))):
        if time.time() > deadline:
            break
        init_score, init_params = candidates[i]
        
        try:
            res = minimize(_compute_score_wrapper, init_params, method='L-BFGS-B',
                           bounds=[(0.1, 2.0), (0.1, 2.0), (0.0, np.pi/2), (-1.0, 1.0)],
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue

    # If no refinement succeeded, use best grid point
    if best_params is None:
        best_params = candidates[0][1]
        
    r1, r2, delta, cx = best_params
    points = _get_points(r1, r2, delta, cx)
    
    # Final sanity check and scaling
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 0)
    D = np.max(dists)
    if D > 0:
        points = points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in points]
