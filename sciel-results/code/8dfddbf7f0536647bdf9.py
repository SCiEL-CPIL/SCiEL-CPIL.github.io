import numpy as np
import scipy.optimize
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

def _is_convex(pts):
    """Check if pts form a strictly convex polygon."""
    n = len(pts)
    crosses = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
    crosses = np.array(crosses)
    if np.any(crosses == 0):
        return False
    if not np.all(crosses > 0) and not np.all(crosses < 0):
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

def _soft_min(scores, temperature=0.01):
    """Smoothed minimum using log-sum-exp."""
    min_val = np.min(scores)
    shifted = scores - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum

def _smooth_objective(coords_flat, temperature=0.01):
    """Smoothed objective: negative soft-min of vertex scores."""
    n = 20
    pts = coords_flat.reshape(n, 2)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 1000.0
    pts_norm = pts_norm / mean_dist
    
    # Check convexity
    if not _is_convex(pts_norm):
        return 1000.0
    
    # Compute vertex scores
    scores = np.array([_vertex_score(pts_norm[i], pts_norm) for i in range(n)])
    
    # Use soft-min to make objective smooth
    soft_min_val = _soft_min(scores, temperature)
    
    return -soft_min_val

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial guess: regular 20-gon
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts, _ = _normalize(best_pts)
    best_score = _min_vertex_score(best_pts)
    
    # Phase 1: L-BFGS-B with smoothed objective
    try:
        coords_flat = best_pts.flatten()
        
        # Multiple temperature values for annealing
        temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for temp in temperatures:
            if time.time() > deadline - 120.0:
                break
            
            def obj(coords_flat):
                return _smooth_objective(coords_flat, temperature=temp)
            
            result = scipy.optimize.minimize(obj, coords_flat, method='L-BFGS-B',
                                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < -best_score + 0.001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_convex(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate
                        coords_flat = candidate.flatten()
                        
    except Exception:
        pass
    
    # Phase 2: Simulated annealing in coordinate space
    try:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.1
        T_min = 0.001
        cooling_rate = 0.95
        max_iter = 5000
        
        for i in range(max_iter):
            if time.time() > deadline - 60.0:
                break
            
            # Random perturbation
            perturbation = np.random.normal(0, T * 0.5, (20, 2))
            candidate = current_pts + perturbation
            
            # Normalize and check validity
            candidate, valid = _normalize(candidate)
            if not valid or not _is_convex(candidate):
                continue
            
            candidate_score = _min_vertex_score(candidate)
            
            # Accept if better, or with probability exp(-delta/T)
            if candidate_score > current_score:
                current_pts = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = candidate.copy()
            else:
                delta = current_score - candidate_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_pts = candidate
                        current_score = candidate_score
            
            T *= cooling_rate
            if T < T_min:
                break
                
    except Exception:
        pass
    
    # Phase 3: Final local refinement with tight tolerance
    try:
        if time.time() < deadline - 30.0:
            coords_flat = best_pts.flatten()
            
            def obj_tight(coords_flat):
                return _smooth_objective(coords_flat, temperature=0.001)
            
            result = scipy.optimize.minimize(obj_tight, coords_flat, method='L-BFGS-B',
                                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < -best_score + 0.0001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_convex(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate
                        
    except Exception:
        pass
    
    return best_pts
