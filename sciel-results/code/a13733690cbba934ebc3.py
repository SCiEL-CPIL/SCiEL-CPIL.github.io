# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score_polygon(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check strict convexity
    # No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                return 0.0
    
    # Every turn between consecutive edges nonzero and of one sign
    signs = []
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return 0.0
        signs.append(np.sign(cross))
    
    if len(set(signs)) != 1:
        return 0.0
    
    # Total turning +-2*pi
    angles = []
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
        # Normalize to [-pi, pi]
        while turn > np.pi:
            turn -= 2*np.pi
        while turn < -np.pi:
            turn += 2*np.pi
        angles.append(turn)
    
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = normalized - normalized[i]
        d = np.linalg.norm(diffs, axis=1)
        d[i] = np.inf  # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        # Take four distances
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        vertex_score = min(1.0, 1.0 / raw)
        
        min_score = min(min_score, vertex_score)
    
    return min_score


def _neg_score(params):
    """Negative score for minimization."""
    angles = params
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    return -_score_polygon(polygon)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum
    vertex equidistance score.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    deadline = time.time() + 600
    
    # Start with regular polygon
    angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # First, compute baseline score
    x0 = np.cos(angles0)
    y0 = np.sin(angles0)
    polygon0 = np.stack([x0, y0], axis=1)
    baseline_score = _score_polygon(polygon0)
    
    best_polygon = polygon0.copy()
    best_score = baseline_score
    
    # Try optimization with different initializations
    # Strategy: optimize the angles directly
    
    # Use scipy minimize with Nelder-Mead (no gradient needed)
    from scipy.optimize import minimize
    
    # Multiple random perturbations of the regular polygon
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        # Create a perturbed initial guess
        if trial == 0:
            angles_init = angles0.copy()
        else:
            # Random perturbation
            perturb = rng.normal(0, 0.1, n)
            angles_init = angles0 + perturb
            # Sort to maintain cyclic order
            angles_init = np.sort(angles_init)
        
        # Convert to optimization: use angles as parameters
        # We need to maintain convexity, so we optimize the angles
        
        # Use a different parameterization: optimize radial distances
        # Actually, let's just optimize the 20 angles
        
        result = minimize(
            _neg_score,
            angles_init,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if time.time() > deadline - 60:
            break
        
        angles_opt = result.x
        # Sort angles to ensure cyclic order
        angles_opt = np.sort(angles_opt)
        
        x_opt = np.cos(angles_opt)
        y_opt = np.sin(angles_opt)
        polygon_opt = np.stack([x_opt, y_opt], axis=1)
        
        score_opt = _score_polygon(polygon_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_polygon = polygon_opt.copy()
    
    return best_polygon


# EVOLVE-BLOCK-END
