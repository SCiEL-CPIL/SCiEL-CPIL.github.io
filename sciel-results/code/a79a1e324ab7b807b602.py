import numpy as np
import time
from scipy.optimize import minimize

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

def _nelder_mead_obj(coords_flat):
    """Objective for Nelder-Mead: negative min vertex score with validity penalty."""
    n = 20
    pts = coords_flat.reshape(n, 2)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 100.0
    pts_norm = pts_norm / mean_dist
    
    # Check convexity - return large penalty if invalid
    if not _is_valid(pts_norm):
        return 100.0
    
    # Compute vertex scores
    score = _min_vertex_score(pts_norm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Regular 20-gon as baseline
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_norm_regular, ok = _normalize(pts_regular)
    if ok and _is_valid(pts_norm_regular):
        best_score = _min_vertex_score(pts_norm_regular)
        best_pts = pts_norm_regular
    else:
        best_score = 0.0
        best_pts = pts_regular
    
    # Multi-restart Nelder-Mead in coordinate space
    np.random.seed(42)
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 120.0:
            break
        
        # Generate initial point: regular polygon with small random perturbation
        if restart == 0:
            init_pts = best_pts.copy()
        else:
            # Create perturbed regular polygon
            angles_init = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            noise_scale = 0.05 + 0.1 * (restart / num_restarts)
            r = 1.0 + np.random.normal(0, noise_scale, n)
            r = np.maximum(r, 0.3)  # Keep positive
            init_pts = np.column_stack([r * np.cos(angles_init), r * np.sin(angles_init)])
            init_pts, valid = _normalize(init_pts)
            if not valid or not _is_valid(init_pts):
                continue
        
        # Run Nelder-Mead
        try:
            coords_flat = init_pts.flatten()
            result = minimize(
                _nelder_mead_obj,
                coords_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            # Check result
            if result.fun < -best_score + 0.0001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_valid(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate.copy()
                        
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    try:
        if time.time() < deadline - 30.0:
            coords_flat = best_pts.flatten()
            result = minimize(
                _nelder_mead_obj,
                coords_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < -best_score + 0.00001:
                candidate = result.x.reshape(20, 2)
                candidate, valid = _normalize(candidate)
                if valid and _is_valid(candidate):
                    score = _min_vertex_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate.copy()
                        
    except Exception:
        pass
    
    return best_pts
