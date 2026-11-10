import numpy as np
import time
from scipy.optimize import minimize

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[i] = np.inf
        # Sort
        dists = np.sort(dists)
        # d_0 ... d_18
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self (closest point is always self, distance 0)
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
    n = len(pts)
    if n != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # Check convexity
    v_curr = pts
    v_next = np.roll(pts, -1, axis=0)
    v_next2 = np.roll(pts, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # All must be same sign and nonzero
    if np.any(cross <= 1e-6):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts / mean_dist

def _param_to_pts(params):
    n = 20
    angles = params[:n]
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective(params):
    pts = _param_to_pts(params)
    pts_norm = _normalize(pts)
    if pts_norm is None:
        return 1000.0
    
    # Convexity check
    v_curr = pts_norm
    v_next = np.roll(pts_norm, -1, axis=0)
    v_next2 = np.roll(pts_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Penalty for non-convexity
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    score = _min_vertex_score(pts_norm)
    return -score + 10.0 * penalty

def objective(params):
    return _objective(params)

def _fourier_param_to_pts(params, n_harmonics=5):
    """
    params: [a0, b0, a1, b1, ..., a_k, b_k, ...]
    We use a base circle of radius 1, plus Fourier perturbations.
    r(theta) = 1 + sum(a_i cos(i*theta) + b_i sin(i*theta))
    We want to ensure r(theta) > 0 and convexity.
    """
    n = 20
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Base radius
    r = np.ones(n)
    
    # Add harmonics
    # params layout:
    # 0: a0 (DC offset, usually kept 0 or small)
    # 1: b0 (DC offset)
    # 2: a1, 3: b1
    # 4: a2, 5: b2
    # ...
    
    for i in range(1, n_harmonics + 1):
        a_i = params[2*i]
        b_i = params[2*i + 1]
        r += a_i * np.cos(i * theta) + b_i * np.sin(i * theta)
        
    # Ensure radius is positive
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _fourier_objective(params, n_harmonics=5):
    pts = _fourier_param_to_pts(params, n_harmonics)
    pts_norm = _normalize(pts)
    if pts_norm is None:
        return 1000.0
    
    # Convexity check
    v_curr = pts_norm
    v_next = np.roll(pts_norm, -1, axis=0)
    v_next2 = np.roll(pts_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Penalty for non-convexity
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    score = _min_vertex_score(pts_norm)
    return -score + 10.0 * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 - 10.0 # Safety margin
    
    n = 20
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    
    best_polygon = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _min_vertex_score(_normalize(best_polygon))
    
    # Strategy 1: Fourier Radial Perturbation
    # This allows for smooth, convex deformations of the circle.
    n_harmonics = 5
    # Parameters: a0, b0, a1, b1, ..., a5, b5
    # Total params: 2 * (n_harmonics + 1)
    # a0, b0 are DC offsets. We can fix a0=0, b0=0 or let them vary slightly.
    # Let's just use a1..a5, b1..b5 to keep it simple and centered.
    # Actually, the function _fourier_param_to_pts uses indices 2*i, 2*i+1 for i in 1..n_h.
    # So we need 2*(n_harmonics) parameters.
    n_params_fourier = 2 * n_harmonics
    x0_fourier = np.zeros(n_params_fourier)
    
    # Run optimization on Fourier params
    try:
        res_fourier = minimize(_fourier_objective, x0_fourier, args=(n_harmonics,), 
                               method='Nelder-Mead', 
                               options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if res_fourier.success or res_fourier.fun < 0: # If it found a valid negative objective (positive score)
            pts_fourier = _fourier_param_to_pts(res_fourier.x, n_harmonics)
            pts_fourier_norm = _normalize(pts_fourier)
            if pts_fourier_norm is not None and _is_valid(pts_fourier_norm):
                score_fourier = _min_vertex_score(pts_fourier_norm)
                if score_fourier > best_score:
                    best_score = score_fourier
                    best_polygon = pts_fourier_norm
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 50:
        return best_polygon

    # Strategy 2: Free Angle/Radius Optimization (Polish)
    # Use the best polygon found so far as initial guess for free optimization
    # Convert best_polygon to angles/radii
    pts = best_polygon
    # Ensure pts are centered
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    radii_init = np.linalg.norm(pts_centered, axis=1)
    angles_init = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
    
    x0_free = np.concatenate([angles_init, radii_init])
    
    try:
        res_free = minimize(objective, x0_free, method='Nelder-Mead', 
                            options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        angles_opt = res_free.x[:20]
        radii_opt = res_free.x[20:]
        
        x = radii_opt * np.cos(angles_opt)
        y = radii_opt * np.sin(angles_opt)
        polygon = np.column_stack((x, y))
        
        # Normalize
        polygon_norm = _normalize(polygon)
        if polygon_norm is not None and _is_valid(polygon_norm):
            score_free = _min_vertex_score(polygon_norm)
            if score_free > best_score:
                best_score = score_free
                best_polygon = polygon_norm
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_polygon

    # Strategy 3: Final Polish with tighter tolerances
    if best_score > 0:
        pts = best_polygon
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        radii_init = np.linalg.norm(pts_centered, axis=1)
        angles_init = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
        x0_polish = np.concatenate([angles_init, radii_init])
        
        try:
            res_polish = minimize(objective, x0_polish, method='Nelder-Mead', 
                                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            angles_opt = res_polish.x[:20]
            radii_opt = res_polish.x[20:]
            
            x = radii_opt * np.cos(angles_opt)
            y = radii_opt * np.sin(angles_opt)
            polygon = np.column_stack((x, y))
            
            polygon_norm = _normalize(polygon)
            if polygon_norm is not None and _is_valid(polygon_norm):
                score_polish = _min_vertex_score(polygon_norm)
                if score_polish > best_score:
                    best_score = score_polish
                    best_polygon = polygon_norm
        except Exception:
            pass

    # Fallback to regular polygon if nothing worked or invalid
    if best_score <= 0 or not _is_valid(best_polygon):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_polygon
