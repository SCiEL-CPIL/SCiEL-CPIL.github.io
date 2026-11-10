import numpy as np
import time
from scipy.optimize import differential_evolution

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

def _param_to_pts(params):
    """
    Convert Fourier coefficients to polygon vertices.
    params[0]: Base radius R
    params[1]: a1 (cos theta)
    params[2]: b1 (sin theta)
    params[3]: a2 (cos 2theta)
    params[4]: b2 (sin 2theta)
    params[5]: a3 (cos 3theta)
    params[6]: b3 (sin 3theta)
    params[7]: a4 (cos 4theta)
    params[8]: b4 (sin 4theta)
    params[9]: a5 (cos 5theta)
    params[10]: b5 (sin 5theta)
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = params[0]
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = params[1:11]
    
    r = R + a1 * np.cos(theta) + b1 * np.sin(theta) + \
            a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta) + \
            a3 * np.cos(3 * theta) + b3 * np.sin(3 * theta) + \
            a4 * np.cos(4 * theta) + b4 * np.sin(4 * theta) + \
            a5 * np.cos(5 * theta) + b5 * np.sin(5 * theta)
    
    # Ensure positive radii to avoid self-intersection near center
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _objective(params):
    pts = _param_to_pts(params)
    norm_pts, valid_norm = _normalize(pts)
    if not valid_norm:
        return 0.0
    if not _is_valid(norm_pts):
        return 0.0
    score = _min_vertex_score(norm_pts)
    return score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial best: Regular polygon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    norm_best, _ = _normalize(best_pts)
    best_score = _min_vertex_score(norm_best)
    
    # Differential Evolution setup
    # Bounds for parameters
    # R is around 1.0 (normalized later, but scale matters for DE)
    # Fourier coefficients should be small to maintain convexity
    bounds = [
        (0.5, 1.5),  # R
        (-0.1, 0.1), # a1
        (-0.1, 0.1), # b1
        (-0.05, 0.05), # a2
        (-0.05, 0.05), # b2
        (-0.02, 0.02), # a3
        (-0.02, 0.02), # b3
        (-0.01, 0.01), # a4
        (-0.01, 0.01), # b4
        (-0.005, 0.005), # a5
        (-0.005, 0.005), # b5
    ]
    
    # Use a seed for reproducibility, but DE is stochastic
    # popsize 15, maxiter 100. 
    # We need to check time inside the callback or just run it and hope it fits.
    # Given the complexity of _min_vertex_score (O(N^2 log N)), 
    # 15 * 100 * 11 evaluations = ~16500 evaluations.
    # Each eval is ~1ms? 16.5s. This is safe.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        if time.time() < deadline:
            trial_pts = _param_to_pts(result.x)
            norm_trial, valid_norm = _normalize(trial_pts)
            if valid_norm and _is_valid(norm_trial):
                trial_score = _min_vertex_score(norm_trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts.copy()
                    
    except Exception as e:
        pass # Fallback to regular polygon if DE fails
    
    # If we have time left, try a second round with tighter bounds around the best found
    if time.time() < deadline - 60:
        # Extract best params if result exists and was good
        if 'result' in locals() and result.success:
            best_params = result.x
            # Tighten bounds
            tight_bounds = []
            for i, (lo, hi) in enumerate(bounds):
                center = best_params[i]
                width = (hi - lo) * 0.1
                tight_bounds.append((max(lo, center - width), min(hi, center + width)))
            
            try:
                result2 = differential_evolution(
                    _objective, 
                    tight_bounds, 
                    seed=43, 
                    maxiter=50, 
                    popsize=10,
                    tol=1e-9
                )
                if time.time() < deadline:
                    trial_pts2 = _param_to_pts(result2.x)
                    norm_trial2, valid_norm2 = _normalize(trial_pts2)
                    if valid_norm2 and _is_valid(norm_trial2):
                        trial_score2 = _min_vertex_score(norm_trial2)
                        if trial_score2 > best_score:
                            best_score = trial_score2
                            best_pts = trial_pts2.copy()
            except Exception:
                pass

    return best_pts
