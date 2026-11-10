# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    tri_a = tri_indices[:, 0]
    tri_b = tri_indices[:, 1]
    tri_c = tri_indices[:, 2]
    
    def compute_score(points):
        """Compute A_min / A_hull ratio."""
        try:
            # Compute all triangle areas
            pa = points[tri_a]
            pb = points[tri_b]
            pc = points[tri_c]
            
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            areas = 0.5 * np.abs(pa[:, 0] * (pb[:, 1] - pc[:, 1]) + 
                                 pb[:, 0] * (pc[:, 1] - pa[:, 1]) + 
                                 pc[:, 0] * (pa[:, 1] - pb[:, 1]))
            
            a_min = np.min(areas)
            
            # Convex hull area
            try:
                hull = ConvexHull(points)
                a_hull = hull.volume
            except Exception:
                return 0.0
            
            if a_hull <= 0 or not np.isfinite(a_hull):
                return 0.0
            
            return a_min / a_hull
        except Exception:
            return 0.0
    
    def soft_min_area(points, beta=50.0):
        """Soft minimum of triangle areas using log-sum-exp."""
        pa = points[tri_a]
        pb = points[tri_b]
        pc = points[tri_c]
        
        areas = 0.5 * np.abs(pa[:, 0] * (pb[:, 1] - pc[:, 1]) + 
                             pb[:, 0] * (pc[:, 1] - pa[:, 1]) + 
                             pc[:, 0] * (pa[:, 1] - pb[:, 1]))
        
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        # This is a smooth approximation to min(areas)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        soft_min = -1.0 / beta * (max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val))))
        
        return soft_min
    
    def objective(x, beta=50.0):
        """Objective function to minimize: -soft_min_area / hull_area."""
        points = x.reshape(n, 2)
        
        # Soft min of triangle areas
        pa = points[tri_a]
        pb = points[tri_b]
        pc = points[tri_c]
        
        areas = 0.5 * np.abs(pa[:, 0] * (pb[:, 1] - pc[:, 1]) + 
                             pb[:, 0] * (pc[:, 1] - pa[:, 1]) + 
                             pc[:, 0] * (pa[:, 1] - pb[:, 1]))
        
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        soft_min = -1.0 / beta * (max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val))))
        
        # Convex hull area
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
        except Exception:
            return 1e10  # Large penalty for invalid configuration
        
        if a_hull <= 0 or not np.isfinite(a_hull):
            return 1e10
        
        return -soft_min / a_hull
    
    # Start with a regular polygon
    angles = 2 * np.pi * np.arange(n) / n
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations and optimization runs
    rng = np.random.default_rng(seed=123)
    
    for run in range(20):
        if time.time() > deadline:
            break
        
        # Different starting points
        if run == 0:
            x = x0.copy()
        elif run == 1:
            # Slightly perturbed regular polygon
            x = x0 + rng.normal(0, 0.05, size=x0.shape)
        elif run == 2:
            # More perturbation
            x = x0 + rng.normal(0, 0.15, size=x0.shape)
        else:
            # Random starting point
            x = rng.uniform(-1, 1, size=x0.shape)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective,
                x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            points = result.x.reshape(n, 2)
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
    
    # Final polish with higher precision
    if best_points is not None and time.time() < deadline - 30:
        try:
            x = best_points.flatten()
            result = minimize(
                objective,
                x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            points = result.x.reshape(n, 2)
            score = compute_score(points)
            if score > best_score:
                best_points = points.copy()
                best_score = score
        except Exception:
            pass
    
    # If no good result, return the regular polygon
    if best_points is None:
        best_points = x0.reshape(n, 2)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = x0.reshape(n, 2)
    
    return best_points

# EVOLVE-BLOCK-END
