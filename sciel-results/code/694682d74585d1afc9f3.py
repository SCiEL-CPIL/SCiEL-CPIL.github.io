# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized to maximize the minimum vertex equidistance score.
    Uses angular perturbation optimization from a regular polygon.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def angles_to_points(delta):
        angles = base_angles + delta
        angles = np.sort(angles)
        angles = angles - angles[0]
        # Ensure within [0, 2pi)
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    
    def compute_min_score(pts):
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        dists_to_centroid = np.sqrt((pts**2).sum(axis=1))
        mean_dist = dists_to_centroid.mean()
        if mean_dist <= 0.01:
            return 0.0
        pts = pts / mean_dist
        
        min_score = 1.0
        for i in range(n):
            vi = pts[i]
            other_pts = np.delete(pts, i, axis=0)
            d = np.sqrt(((other_pts - vi)**2).sum(axis=1))
            d_sorted = np.sort(d)
            
            if d_sorted[0] < 0.05:
                return 0.0
            
            gaps = np.diff(d_sorted)
            k = int(np.argmin(gaps))
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                return 0.0
            
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k-1:k+3]
            
            ratios = np.maximum(selected / D, D / selected)
            raw = ratios.mean()
            score = min(1.0, 1.0 / raw)
            if score < min_score:
                min_score = score
        
        return min_score
    
    def neg_score(delta):
        pts = angles_to_points(delta)
        s = compute_min_score(pts)
        return -s
    
    delta0 = np.zeros(n)
    
    result = minimize(neg_score, delta0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    
    delta_opt = result.x
    pts = angles_to_points(delta_opt)
    
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    dists_to_centroid = np.sqrt((pts**2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        pts = pts / mean_dist
    
    return pts


# EVOLVE-BLOCK-END
