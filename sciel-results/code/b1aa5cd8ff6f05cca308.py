# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Uses Nelder-Mead optimization on Cartesian coordinates to maximize
    the minimum vertex score for a convex 20-gon.
    """
    n = 20
    
    def compute_metric(points):
        """
        Computes the metric (minimum vertex score) for a given set of points.
        Returns 0 if invalid.
        """
        # Check for repeated points
        # Vectorized distance check
        d_mat = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(d_mat, 1.0)
        if np.any(d_mat < 1e-6):
            return 0.0

        # Check strict convexity
        # Compute cross products of consecutive edges
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # All cross products must be non-zero and same sign
        if np.any(np.abs(cross) < 1e-9):
            return 0.0
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return 0.0

        # Normalize
        centroid = np.mean(points, axis=0)
        points_norm = points - centroid
        dists_to_centroid = np.linalg.norm(points_norm, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        points_norm = points_norm / mean_dist
        
        # Calculate vertex scores
        min_score = 1.0
        for i in range(n):
            v = points_norm[i]
            # Distances to all other vertices
            d_all = np.linalg.norm(points_norm - v, axis=1)
            d_all[i] = np.inf # Exclude self
            
            # Sort the 19 distances
            d_sorted = np.sort(d_all)
            
            # Check if any distance < 0.05
            if d_sorted[0] < 0.05:
                return 0.0 # Vertex scores 0, so min score is 0
            
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(gaps)
            
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                return 0.0
            
            # Select 4 distances
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k-1:k+3]
                
            # Calculate raw score
            # raw = mean of max(d/D, D/d)
            # Ensure no division by zero (D > 0.1, d > 0.05, so safe)
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            
            # Vertex score = min(1, 1/raw)
            score = min(1.0, 1.0 / raw)
            
            if score < min_score:
                min_score = score
                
        return min_score

    def loss(coords):
        """
        Loss function for optimization.
        coords is a flat array of shape (40,).
        """
        points = coords.reshape(n, 2)
        
        # Check convexity for penalty
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # Penalty for non-convexity
        # If any cross product is near zero or signs differ, penalize heavily
        if np.any(np.abs(cross) < 1e-6):
            return 1e6
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return 1e6
            
        # Check for repeated points
        d_mat = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(d_mat, 1.0)
        if np.any(d_mat < 1e-6):
            return 1e6

        metric = compute_metric(points)
        # We want to maximize metric.
        # If metric is 0 (invalid or bad), return high loss.
        # If metric is 1.0, return 0.
        if metric == 0.0:
             return 10.0
        return 1.0 - metric

    # Initialize with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    init_points = np.stack([x, y], axis=1)
    init_coords = init_points.flatten()
    
    # Run Nelder-Mead
    # options: maxiter, xatol, fatol
    result = minimize(loss, init_coords, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
    
    optimal_coords = result.x
    points = optimal_coords.reshape(n, 2)
    
    # Verify and normalize
    metric = compute_metric(points)
    
    # Fallback to regular polygon if optimization failed
    if metric < 0.9:
        points = init_points
        
    # Final normalization
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return init_points
        
    points = points / mean_dist
    
    return points

# EVOLVE-BLOCK-END
