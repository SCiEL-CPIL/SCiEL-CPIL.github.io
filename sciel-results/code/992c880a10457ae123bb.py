import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, ConvexHull.volume gives area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(pts)
    ch_area = _convex_hull_area(pts)
    if ch_area <= 1e-12:
        return 0.0
    return min_ta / ch_area

def _smooth_min_triangle_area(pts, beta=1.0):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    Returns a smooth function of the points.
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Log-sum-exp trick for stability
    max_area = np.max(areas)
    log_sum_exp = max_area + np.log(np.sum(np.exp(-(areas - max_area) / beta)))
    # The smooth min is approximately -beta * log_sum_exp + max_area * (1 - beta) ... 
    # Actually, the smooth min is: -beta * log(sum(exp(-area_i / beta)))
    # Let's compute that directly
    exp_vals = np.exp(-areas / beta)
    log_sum = np.log(np.sum(exp_vals))
    smooth_min = -beta * log_sum
    return smooth_min

def _neg_smooth_score(x, beta=1.0):
    """
    Objective function for optimization.
    x: 2N coordinates of N points.
    Returns negative of the smooth score (min triangle area / hull area).
    """
    pts = x.reshape(-1, 2)
    
    # Normalize: translate to centroid, scale so hull area = 1
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 1e10  # Large penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    pts = pts * scale
    
    # Now hull area is 1
    smooth_min_ta = _smooth_min_triangle_area(pts, beta)
    
    # Score is smooth_min_ta / 1 = smooth_min_ta
    # We want to maximize this, so minimize the negative
    return -smooth_min_ta

def _gradient_neg_smooth_score(x, beta=1.0):
    """
    Numerical gradient of the negative smooth score.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _neg_smooth_score(x, beta)
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        f_pert = _neg_smooth_score(x_pert, beta)
        grad[i] = (f_pert - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initial configurations to try
    def _regular_polygon(n, jitter=0.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        if jitter > 0:
            pts += np.random.normal(0, jitter, size=pts.shape)
        return pts
    
    def _random_convex_polygon(n):
        # Generate random angles and sort them
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n))
        # Random radii
        radii = np.random.uniform(0.5, 1.0, n)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        return pts
    
    # Try multiple restarts
    num_restarts = 20
    beta = 0.1  # Temperature for smooth approximation
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose initial configuration
        if restart % 3 == 0:
            x0 = _regular_polygon(N, jitter=0.1).flatten()
        elif restart % 3 == 1:
            x0 = _random_convex_polygon(N).flatten()
        else:
            x0 = _regular_polygon(N, jitter=0.05).flatten()
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _neg_smooth_score,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=_gradient_neg_smooth_score,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                # Reconstruct points
                pts = result.x.reshape(-1, 2)
                
                # Normalize: translate to centroid, scale so hull area = 1
                centroid = np.mean(pts, axis=0)
                pts = pts - centroid
                hull_area = _convex_hull_area(pts)
                if hull_area > 1e-12:
                    scale = 1.0 / np.sqrt(hull_area)
                    pts = pts * scale
                
                # Compute exact score
                score = _score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
        except Exception as e:
            pass
    
    # If we still have time, try a few more with different betas
    if time.time() < deadline - 20:
        for beta in [0.05, 0.2]:
            if time.time() > deadline - 10:
                break
            x0 = _regular_polygon(N, jitter=0.05).flatten()
            try:
                result = minimize(
                    _neg_smooth_score,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_gradient_neg_smooth_score,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_score:
                    pts = result.x.reshape(-1, 2)
                    centroid = np.mean(pts, axis=0)
                    pts = pts - centroid
                    hull_area = _convex_hull_area(pts)
                    if hull_area > 1e-12:
                        scale = 1.0 / np.sqrt(hull_area)
                        pts = pts * scale
                    score = _score(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N)
        centroid = np.mean(best_pts, axis=0)
        best_pts = best_pts - centroid
        hull_area = _convex_hull_area(best_pts)
        if hull_area > 1e-12:
            scale = 1.0 / np.sqrt(hull_area)
            best_pts = best_pts * scale
    
    return best_pts
