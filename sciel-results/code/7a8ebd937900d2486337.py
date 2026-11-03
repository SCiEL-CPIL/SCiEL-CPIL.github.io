"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Start with a regular 13-gon (known to be a strong candidate for maximizing
       minimum triangle area among convex position points).
    2. Refine using local optimization: maximize the minimum triangle area
       subject to points remaining in convex position.
    3. Normalize by convex hull area.
    """
    
    # Generate regular 13-gon centered at origin
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to compute convex hull area
    def convex_hull_area(pts):
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume is area
        except:
            return 1e-10
    
    # Function to compute minimum triangle area among all triples
    def min_triangle_area(pts):
        n = len(pts)
        min_area = float('inf')
        for i, j, k in combinations(range(n), 3):
            # Area of triangle with vertices pts[i], pts[j], pts[k]
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area
    
    # Objective: maximize min triangle area / convex hull area
    # We minimize the negative of this ratio
    def objective(x):
        pts = x.reshape(N, 2)
        
        # Check convex position: all points should be on convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            # Check if all points are on the hull
            if len(hull.vertices) != N:
                # Penalize non-convex positions
                return -1e6
        except:
            return -1e6
        
        hull_area = convex_hull_area(pts)
        if hull_area < 1e-10:
            return -1e6
        
        mt_area = min_triangle_area(pts)
        
        # We want to maximize mt_area / hull_area, so minimize negative
        return -(mt_area / hull_area)
    
    # Gradient-free optimization using Nelder-Mead
    # Start from regular polygon
    x0 = points.flatten()
    
    # Use Nelder-Mead for robustness
    result = minimize(
        objective, 
        x0, 
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    optimized_points = result.x.reshape(N, 2)
    
    # If optimization failed badly, fall back to regular polygon
    if result.fun < -1e5:
        optimized_points = points
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = convex_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        # Center
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
