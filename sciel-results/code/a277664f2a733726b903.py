# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _smooth_score(p):
    """
    Smooth surrogate for optimization.
    Penalizes non-convexity and low vertex scores.
    """
    pn, _, _ = _normalize(p)
    if pn is None:
        return -100.0
    
    # Check convexity with soft penalty
    n = len(pn)
    cross_products = []
    for i in range(n):
        p0 = pn[i]
        p1 = pn[(i + 1) % n]
        p2 = pn[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
        
    cross_arr = np.array(cross_products)
    if np.all(cross_arr > 0):
        convexity_penalty = 0.0
    elif np.all(cross_arr < 0):
        convexity_penalty = 0.0
    else:
        # Mixed signs or zero
        convexity_penalty = -10.0 * np.std(cross_arr) - 5.0 * np.min(np.abs(cross_arr))
        
    # Vertex scores
    min_vs = 1.0
    for i in range(n):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
            
    # Smooth the min to make it differentiable-ish
    # Use a soft-min approximation
    scores = np.array([_vertex_score(pn, i) for i in range(n)])
    temp = 0.1
    soft_min = -temp * np.log(np.sum(np.exp(-scores / temp)))
    
    return soft_min + convexity_penalty

def _project_to_convex(pts):
    """
    Simple heuristic to project points towards convexity.
    If non-convex, try to pull inward or outward.
    """
    pn, _, _ = _normalize(pts)
    if pn is None:
        return pts
        
    n = len(pn)
    # Check convexity
    is_convex = _check_strictly_convex(pn)
    if is_convex:
        return pn
        
    # If not convex, try to fix by scaling or moving points
    # This is a crude projection. We'll rely on DE to find convex regions.
    # For now, just return the normalized points.
    return pn

def equidistant_points_polygon20() -> np.ndarray:
    """
    Direct 2D coordinate optimization using differential evolution.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    n = 20
    
    # Bounds for x and y coordinates
    # Initial guess: regular polygon of radius 1
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    
    # Flatten to 40D vector
    x_flat = np.zeros(2 * n)
    x_flat[0::2] = x0
    x_flat[1::2] = y0
    
    # Define bounds
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    def objective(x):
        if time.time() > deadline:
            return -100.0
        x = np.array(x)
        pts = np.zeros((n, 2))
        pts[:, 0] = x[0::2]
        pts[:, 1] = x[1::2]
        
        # Use smooth score for optimization
        score = _smooth_score(pts)
        return -score  # Minimize negative score
    
    # Run differential evolution
    # We need to be careful with the time budget.
    # DE can be slow. Let's limit maxiter and popsize.
    # popsize=15, maxiter=100 is a reasonable start.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=0.01, 
            mutation=(0.5, 1.5), 
            recombination=0.7
        )
        
        best_x = result.x
        best_score = -result.fun
        
        # Refine with local search (Nelder-Mead or similar) if time permits
        if time.time() < deadline - 60:
            from scipy.optimize import minimize
            
            # Use Nelder-Mead for local refinement
            res = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if res.fun < result.fun:
                best_x = res.x
                best_score = -res.fun
                
    except Exception as e:
        # Fallback to regular polygon if optimization fails
        best_x = x_flat.copy()
        best_score = 0.0
        
    # Convert back to points
    best_pts = np.zeros((n, 2))
    best_pts[:, 0] = best_x[0::2]
    best_pts[:, 1] = best_x[1::2]
    
    # Final normalization and check
    pn, _, _ = _normalize(best_pts)
    if pn is None:
        # Fallback
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    if not _check_strictly_convex(pn):
        # If still not convex, try a quick repair or fallback
        # For now, return the best convex polygon found during optimization
        # or fallback to regular
        # Let's try to find a convex subset or just return regular
        # A better fallback: use the best score polygon from DE that was convex
        # But we didn't track that. So we return regular.
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return pn

# EVOLVE-BLOCK-END
