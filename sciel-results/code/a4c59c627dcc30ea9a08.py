# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Approach:
    - Start with a regular 13-gon as the hull.
    - Place interior points via a Delaunay-based Lloyd relaxation to uniformize density.
    - Optimize all points jointly to maximize the minimum triangle area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Initial configuration: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # We'll treat all points as free to move, but we want to encourage a convex hull.
    # Strategy: 
    # 1. First, do a few iterations of Lloyd's algorithm on a subset to get a good interior layout.
    #    But since n=13 is small, let's just use a heuristic initial layout:
    #    - 7 points on the hull (regular heptagon)
    #    - 6 points inside arranged in a hexagonal pattern
    
    # Let's try a better initial guess:
    # Outer ring: 7 points on a circle
    n_outer = 7
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_points = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner ring: 6 points on a smaller circle, offset
    n_inner = 6
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / 6
    inner_radius = 0.5
    inner_points = np.column_stack([inner_radius * np.cos(angles_inner), inner_radius * np.sin(angles_inner)])
    
    points = np.vstack([outer_points, inner_points])
    
    # Compute all triangle areas
    def min_triangle_area(pts):
        """Compute the minimum area among all triangles formed by 3 points."""
        n_pts = len(pts)
        min_area = np.inf
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 0.0
    
    def objective(x):
        """Negative of the ratio A_min / A_hull (we minimize the negative)."""
        pts = x.reshape(-1, 2)
        # Check for NaN or inf
        if not np.all(np.isfinite(pts)):
            return 1e10
        a_min = min_triangle_area(pts)
        a_hull = hull_area(pts)
        if a_hull <= 0:
            return 1e10
        return -(a_min / a_hull)
    
    # Optimize using Nelder-Mead
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    points_opt = result.x.reshape(-1, 2)
    
    # Verify the result is valid
    if not np.all(np.isfinite(points_opt)):
        # Fall back to initial points
        points_opt = points
    
    # Ensure the convex hull is well-defined
    try:
        hull = ConvexHull(points_opt)
    except:
        # If hull fails, use initial points
        points_opt = points
    
    return points_opt


# EVOLVE-BLOCK-END
