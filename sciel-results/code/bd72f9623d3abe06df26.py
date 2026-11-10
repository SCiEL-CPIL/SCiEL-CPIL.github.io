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

def _fourier_to_pts(coeffs):
    """
    Convert Fourier coefficients to polygon vertices.
    coeffs: array of shape (N,) where N is even.
    coeffs[0] is ignored (DC term handled by normalization).
    coeffs[1], coeffs[2] are cos/sin for k=1
    coeffs[3], coeffs[4] are cos/sin for k=2
    ...
    We use k=1 to k=K.
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Base circle
    r = np.ones(n)
    
    # Add harmonic components
    # coeffs[1:2*K+1]
    # Let's say we use up to K=5 harmonics
    K = len(coeffs) // 2
    for k in range(1, K + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
        
    # Ensure r > 0 to avoid weird shapes, though normalization helps
    # If r becomes negative, the point flips, which might break convexity or order.
    # For a convex polygon containing the origin, r(theta) > 0 is required.
    # We'll add a penalty in the objective if r <= 0.
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y])
    return pts

def _objective(params):
    """
    Objective function for optimization.
    We want to maximize the minimum vertex score.
    So we minimize -min_vertex_score.
    Add penalties for invalid shapes (non-convex, self-intersecting, etc.)
    """
    # params: [a1, b1, a2, b2, ..., aK, bK]
    # We fix the base radius to 1.0 in _fourier_to_pts
    pts = _fourier_to_pts(params)
    
    # Check validity and convexity
    if not _is_valid(pts):
        return 1000.0 # Large penalty
    
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1000.0
    
    score = _min_vertex_score(norm_pts)
    
    # Penalty if any radius component is too small (risk of non-convexity or self-intersection)
    # The radius r(theta) = 1 + sum(...)
    # If min(r) < 0.5, it's getting close to the center, which might cause issues with the "D < 0.1" check or convexity.
    # Actually, _is_valid checks convexity.
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    # Start with regular polygon (all Fourier coefficients 0)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    norm_pts_reg, valid_reg = _normalize(pts_reg)
    
    best_score = 0.0
    if valid_reg and _is_valid(norm_pts_reg):
        best_score = _min_vertex_score(norm_pts_reg)
    best_pts = pts_reg.copy()
    
    # Optimization strategy:
    # Use Nelder-Mead on the Fourier coefficients.
    # We will try a few different sets of harmonics (K=2, K=3, K=4) to see if higher frequencies help.
    # Regular polygon is K=0.
    
    rng = np.random.default_rng(42)
    
    for K in [2, 3, 4, 5]:
        if time.time() > deadline - 60:
            break
            
        # Initial guess: small random perturbations
        # We need 2*K parameters
        initial_params = rng.normal(0, 0.1, size=2 * K)
        
        # Run Nelder-Mead
        try:
            res = minimize(_objective, initial_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < -best_score: # i.e., -score < -best_score => score > best_score
                optimal_params = res.x
                pts_opt = _fourier_to_pts(optimal_params)
                if _is_valid(pts_opt):
                    norm_pts_opt, valid_opt = _normalize(pts_opt)
                    if valid_opt:
                        score_opt = _min_vertex_score(norm_pts_opt)
                        if score_opt > best_score:
                            best_score = score_opt
                            best_pts = pts_opt.copy()
        except Exception:
            pass
            
        # Also try a second random start for this K
        if time.time() > deadline - 60:
            break
        initial_params2 = rng.normal(0, 0.05, size=2 * K)
        try:
            res2 = minimize(_objective, initial_params2, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < -best_score:
                optimal_params2 = res2.x
                pts_opt2 = _fourier_to_pts(optimal_params2)
                if _is_valid(pts_opt2):
                    norm_pts_opt2, valid_opt2 = _normalize(pts_opt2)
                    if valid_opt2:
                        score_opt2 = _min_vertex_score(norm_pts_opt2)
                        if score_opt2 > best_score:
                            best_score = score_opt2
                            best_pts = pts_opt2.copy()
        except Exception:
            pass

    return best_pts
