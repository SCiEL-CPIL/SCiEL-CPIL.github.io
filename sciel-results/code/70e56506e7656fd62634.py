# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    def triangle_area(p1, p2, p3):
        """Compute the area of a triangle given three points."""
        return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
    
    def min_triangle_area(points):
        """Compute the minimum triangle area among all triples of points."""
        min_area = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(points):
        """Compute the area of the convex hull of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def objective(x):
        """Objective function to minimize: negative of the ratio A_min / A_hull."""
        points = x.reshape(n, 2)
        
        # Check for invalid configurations
        if np.any(~np.isfinite(points)):
            return 1e10
        
        # Compute hull area
        a_hull = hull_area(points)
        if a_hull < 1e-10:
            return 1e10
        
        # Compute min triangle area
        a_min = min_triangle_area(points)
        
        # Return negative ratio (we want to maximize)
        return -(a_min / a_hull)
    
    # Start with a regular polygon arrangement as initial guess
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Use Nelder-Mead optimization to find better arrangement
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    points = result.x.reshape(n, 2)
    
    # Ensure points are finite
    if not np.all(np.isfinite(points)):
        # Fallback to random points if optimization failed
        rng = np.random.default_rng(seed=42)
        points = rng.random((n, 2))
    
    return points


# EVOLVE-BLOCK-END
