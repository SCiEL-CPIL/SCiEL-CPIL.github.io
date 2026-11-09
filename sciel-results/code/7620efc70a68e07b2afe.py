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

def _param_to_pts(params):
    # params: 20 radii, 20 angles (relative to regular)
    n = 20
    radii = params[:n]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + params[n:]
    # Ensure angles are sorted? The parametrization keeps them in order if perturbations are small.
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _objective(params):
    pts = _param_to_pts(params)
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e6
    if not _is_valid(norm_pts):
        return 1e6
    score = _min_vertex_score(norm_pts)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0

    n = 20
    # Initial guess: Regular polygon
    # Radii = 1, Angle offsets = 0
    x0 = np.zeros(2*n)
    x0[:n] = 1.0
    
    best_score = -1.0
    best_pts = None
    
    # Get initial score
    init_pts = _param_to_pts(x0)
    norm_init, v = _normalize(init_pts)
    if v and _is_valid(norm_init):
        best_score = _min_vertex_score(norm_init)
        best_pts = init_pts.copy()
    else:
        best_score = 0.0
        best_pts = init_pts.copy()

    # Optimization
    if time.time() < deadline - 60:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success or res.fun < _objective(x0):
                trial_pts = _param_to_pts(res.x)
                norm_trial, v = _normalize(trial_pts)
                if v and _is_valid(norm_trial):
                    trial_score = _min_vertex_score(norm_trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial_pts.copy()
        except Exception:
            pass

    # Fallback random search if optimization didn't improve or failed
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Small perturbations to radii and angles
            radii = 1.0 + rng.normal(0, 0.1, size=n)
            angles = rng.normal(0, 0.1, size=n)
            params = np.concatenate([radii, angles])
            pts = _param_to_pts(params)
            norm_pts, v = _normalize(pts)
            if not v or not _is_valid(norm_pts):
                continue
            score = _min_vertex_score(norm_pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()

    return best_pts
