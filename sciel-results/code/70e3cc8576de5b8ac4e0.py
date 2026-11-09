# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_strictly_convex(poly):
    """Check if polygon is strictly convex (no repeated points, all turns same sign, nonzero)."""
    n = len(poly)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(poly[i] - poly[j]) < 1e-12:
                return False
    # Compute cross products of consecutive edges
    turns = []
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        turns.append(cross)
    turns = np.array(turns)
    # All turns must be nonzero and of the same sign
    if np.any(np.abs(turns) < 1e-12):
        return False
    if np.any(turns > 0) and np.any(turns < 0):
        return False
    # Total turning should be +/- 2*pi (check by summing angles)
    # For a strictly convex polygon, total turning is 2*pi
    return True


def _score_polygon(poly):
    """Compute the minimum vertex score over all 20 vertices. Returns 0 if invalid."""
    if not _is_strictly_convex(poly):
        return 0.0
    
    n = len(poly)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(poly, axis=0)
    poly_norm = poly - centroid
    dists_to_centroid = np.linalg.norm(poly_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    poly_norm = poly_norm / mean_dist
    
    min_score = 1.0
    for v_idx in range(n):
        v = poly_norm[v_idx]
        # Compute distances from v to all other vertices
        dists = []
        for j in range(n):
            if j != v_idx:
                d = np.linalg.norm(v - poly_norm[j])
                dists.append(d)
        dists = np.array(dists)
        dists_sorted = np.sort(dists)
        
        # Check if any distance < 0.05
        if np.any(dists_sorted < 0.05):
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists_sorted)
        k = np.argmin(diffs)
        D = (dists_sorted[k] + dists_sorted[k + 1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            break
        
        # Take four distances
        if k == 0:
            four_dists = dists_sorted[0:4]
        elif k == 17:
            four_dists = dists_sorted[15:19]
        else:
            four_dists = dists_sorted[k - 1:k + 3]
        
        raw = np.mean([max(d / D, D / d) for d in four_dists])
        vertex_score = min(1.0, 1.0 / raw)
        min_score = min(min_score, vertex_score)
    
    return min_score


def _neg_score(params):
    """Negative score for minimization."""
    poly = params.reshape(20, 2)
    score = _score_polygon(poly)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices optimized for the equidistance score.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s deadline
    
    # Start with regular polygon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    best_poly = np.stack([x, y], axis=1)
    best_score = _score_polygon(best_poly)
    
    # Try to improve with local optimization
    # We'll optimize the angles (keeping points on a circle) first
    # Then try general coordinate optimization
    
    # Phase 1: Optimize angles on a circle
    def _neg_score_angles(theta_params):
        theta = theta_params
        # Ensure angles are in [0, 2*pi) and sorted
        theta = np.sort(theta)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        poly = np.stack([x, y], axis=1)
        return -_score_polygon(poly)
    
    # Initial angles
    init_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Try a few random restarts for angle optimization
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            theta0 = init_angles.copy()
        else:
            theta0 = init_angles + np.random.normal(0, 0.1, n)
        
        try:
            result = minimize(_neg_score_angles, theta0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < -best_score:
                theta_opt = result.x
                theta_opt = np.sort(theta_opt)
                x = radius * np.cos(theta_opt)
                y = radius * np.sin(theta_opt)
                poly = np.stack([x, y], axis=1)
                score = _score_polygon(poly)
                if score > best_score:
                    best_score = score
                    best_poly = poly
        except Exception:
            continue
    
    # Phase 2: General coordinate optimization
    # Use the best polygon as starting point
    for restart in range(5):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            params0 = best_poly.flatten()
        else:
            params0 = best_poly.flatten() + np.random.normal(0, 0.05, 40)
        
        try:
            result = minimize(_neg_score, params0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < -best_score:
                poly = result.x.reshape(20, 2)
                score = _score_polygon(poly)
                if score > best_score:
                    best_score = score
                    best_poly = poly
        except Exception:
            continue
    
    return best_poly


# EVOLVE-BLOCK-END
