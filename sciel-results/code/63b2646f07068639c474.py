import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(np.dot(normal, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Repulsion Relaxation (Thomson Problem solver)
    # Minimize sum 1/||p_i - p_j||
    # This spreads points evenly, minimizing covering radius.
    
    alpha = 0.1
    for it in range(1000):
        # Calculate forces
        # diff[i,j] = p_i - p_j
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= 0.998
        
    return points
