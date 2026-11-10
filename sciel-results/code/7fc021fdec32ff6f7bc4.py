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

def _soft_min(scores, beta=100.0):
    """
    Compute a smooth approximation of the minimum of scores using log-sum-exp.
    min(scores) ≈ -1/beta * log(sum(exp(-beta * scores)))
    """
    m = np.min(scores)
    if m == 0:
        return 0.0
    # Shift for numerical stability
    shifted = scores - m
    sum_exp = np.sum(np.exp(-beta * shifted))
    if sum_exp == 0:
        return 0.0
    return m - (1.0 / beta) * np.log(sum_exp)

def _objective_free(params):
    """
    Objective function for free coordinate optimization.
    params: array of shape (40,) representing 20 (x,y) coordinates.
    We minimize -soft_min(scores) + penalty.
    """
    pts = params.reshape(20, 2)
    
    # Check validity (convexity, no repeated points)
    if not _is_valid(pts):
        return 1000.0
        
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1000.0
        
    # Compute scores
    scores = np.array([_vertex_score(norm_pts[i], norm_pts) for i in range(20)])
    
    # Use soft-min to smooth the objective
    min_score = _soft_min(scores, beta=50.0)
    
    # If any score is 0, the soft-min will be small, but we want to avoid 0 scores.
    # The penalty for invalid shapes is already handled.
    # We want to maximize min_score, so we minimize -min_score.
    
    return -min_score

def _objective_grad(params, grad):
    """
    Objective function with gradient for L-BFGS-B.
    Note: Computing the gradient of the soft-min through the discrete sorting
    and selection logic is complex. We will use numerical gradients or 
    finite differences if analytic gradient is too hard.
    For now, let's try to use L-BFGS-B with numerical gradients (default).
    """
    return _objective_free(params)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    # Start with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    norm_pts_reg, valid_reg = _normalize(pts_reg)
    
    best_score = 0.0
    if valid_reg and _is_valid(norm_pts_reg):
        best_score = _min_vertex_score(norm_pts_reg)
    best_pts = pts_reg.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Use L-BFGS-B on free coordinates starting from regular polygon.
    # 2. Use multiple random restarts.
    # 3. Use a soft-min surrogate to make the objective smooth.
    
    # Initial params: regular polygon flattened
    initial_params = pts_reg.flatten()
    
    # Run L-BFGS-B
    try:
        res = minimize(_objective_free, initial_params, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-5, 'ftol': 1e-5})
        
        if res.fun < -best_score:
            pts_opt = res.x.reshape(20, 2)
            if _is_valid(pts_opt):
                norm_pts_opt, valid_opt = _normalize(pts_opt)
                if valid_opt:
                    score_opt = _min_vertex_score(norm_pts_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_pts = pts_opt.copy()
    except Exception:
        pass
    
    # Random restarts with perturbations
    for restart in range(10):
        if time.time() > deadline - 60:
            break
            
        # Perturb the regular polygon
        perturb_scale = 0.1 + restart * 0.05
        perturbations = rng.normal(0, perturb_scale, size=(20, 2))
        initial_params = (pts_reg + perturbations).flatten()
        
        try:
            res = minimize(_objective_free, initial_params, method='L-BFGS-B',
                           options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-5})
            
            if res.fun < -best_score:
                pts_opt = res.x.reshape(20, 2)
                if _is_valid(pts_opt):
                    norm_pts_opt, valid_opt = _normalize(pts_opt)
                    if valid_opt:
                        score_opt = _min_vertex_score(norm_pts_opt)
                        if score_opt > best_score:
                            best_score = score_opt
                            best_pts = pts_opt.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead on the same initial point as a backup
        if time.time() > deadline - 60:
            break
        try:
            res_nm = minimize(_objective_free, initial_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_nm.fun < -best_score:
                pts_opt_nm = res_nm.x.reshape(20, 2)
                if _is_valid(pts_opt_nm):
                    norm_pts_opt_nm, valid_opt_nm = _normalize(pts_opt_nm)
                    if valid_opt_nm:
                        score_opt_nm = _min_vertex_score(norm_pts_opt_nm)
                        if score_opt_nm > best_score:
                            best_score = score_opt_nm
                            best_pts = pts_opt_nm.copy()
        except Exception:
            pass
            
    return best_pts
