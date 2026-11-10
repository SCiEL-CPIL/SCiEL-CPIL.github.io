import numpy as np
import time

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _objective(params):
    """Objective function for optimization. Returns negative min vertex score."""
    pts = params.reshape(20, 2)
    if not _is_valid(pts):
        return 1000.0 # Penalty for invalid
    pts_norm, ok = _normalize(pts)
    if not ok:
        return 1000.0
    score = _min_vertex_score(pts_norm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    start_time = time.time()
    best_pts = base_pts
    best_score = -_objective(base_pts.flatten())
    
    # If base score is already high, we are good.
    # But base is ~0.67. We want > 0.9.
    
    from scipy.optimize import minimize
    
    # Try multiple restarts with different perturbations
    np.random.seed(0)
    for i in range(10):
        if time.time() - start_time > 500:
            break
            
        # Perturb the base polygon
        noise_level = 0.1 * (1 + i) # Increase noise for later restarts
        pts = base_pts + np.random.normal(0, noise_level, base_pts.shape)
        
        try:
            res = minimize(_objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_pts = res.x.reshape(20, 2)
                
                # If we found a very good score, polish it
                if best_score > 0.9:
                    res2 = minimize(_objective, best_pts.flatten(), method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                    if -res2.fun > best_score:
                        best_score = -res2.fun
                        best_pts = res2.x.reshape(20, 2)
                        
        except Exception:
            continue

    final_pts, _ = _normalize(best_pts)
    return final_pts
