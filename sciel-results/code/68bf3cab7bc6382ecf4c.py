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
    params[1:11]: a1, b1, a2, b2, a3, b3, a4, b4, a5, b5
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = params[0]
    coeffs = params[1:11]
    
    r = R * np.ones(n)
    for i in range(5):
        a = coeffs[2*i]
        b = coeffs[2*i+1]
        k = i + 1
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
        
    # Ensure positive radius to avoid singularity at center, though normalization handles scale
    # But for validity, we need distinct points.
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _objective(params, deadline):
    if time.time() > deadline:
        return 1e10
    
    pts = _param_to_pts(params)
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e10
    
    # Check convexity and get penalty
    if not _is_valid(norm_pts):
        # Calculate a penalty based on how much it violates convexity
        # Simpler: just return a large penalty
        return 1e5 + (1.0 - _min_vertex_score(norm_pts))
    
    score = _min_vertex_score(norm_pts)
    return 1.0 - score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial best: Regular 20-gon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    norm_reg, _ = _normalize(pts_reg)
    best_score = _min_vertex_score(norm_reg)
    best_pts = pts_reg.copy()
    
    # Define bounds for Differential Evolution
    # R is roughly 1. Coefficients should be small to maintain convexity and close to regular.
    # If coefficients are too large, the polygon becomes non-convex or self-intersecting.
    # Let's bound R in [0.5, 1.5] and coeffs in [-0.2, 0.2] initially.
    bounds = [(0.5, 1.5)] + [(-0.2, 0.2)] * 10
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            _objective,
            bounds,
            args=(deadline,),
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-5,
            init='latinhypercube'
        )
        
        if res.success or res.fun < 1.0: # If it found something better than penalty
            trial_pts = _param_to_pts(res.x)
            norm_trial, valid = _normalize(trial_pts)
            if valid and _is_valid(norm_trial):
                trial_score = _min_vertex_score(norm_trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts
    except Exception:
        pass
    
    # If DE didn't improve, or if we have time, try a local polish around the best found
    # using coordinate-space Nelder-Mead or similar, but DE is usually good enough for this low-dim space.
    
    return best_pts
