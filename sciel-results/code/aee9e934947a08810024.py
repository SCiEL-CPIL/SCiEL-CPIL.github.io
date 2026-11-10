# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    
    Approach:
    - Fix the hull as a regular polygon (7 points)
    - Use Lloyd's algorithm for interior points (Voronoi centroid iteration)
    - This enforces uniform density which maximizes minimum triangle area
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    n_hull = 7
    n_interior = n - n_hull  # 6 interior points
    
    # Step 1: Create regular heptagon as the hull
    angles_hull = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    hull_radius = 1.0
    hull_points = np.column_stack([hull_radius * np.cos(angles_hull), 
                                    hull_radius * np.sin(angles_hull)])
    
    # Step 2: Initialize interior points with a hexagonal pattern
    angles_inner = np.linspace(0, 2 * np.pi, n_interior, endpoint=False) + np.pi / n_interior
    inner_radius = 0.5
    interior_points = np.column_stack([inner_radius * np.cos(angles_inner), 
                                        inner_radius * np.sin(angles_inner)])
    
    points = np.vstack([hull_points, interior_points])
    
    # Step 3: Lloyd's algorithm for interior points
    for iteration in range(100):
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            break
        
        # For each interior point, find its Voronoi cell and compute centroid
        new_interior = np.zeros_like(interior_points)
        
        for i in range(n_interior):
            idx = n_hull + i
            # Get the Voronoi region for this point
            region_idx = vor.point_region[idx]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - use the point itself
                new_interior[i] = interior_points[i]
                continue
            
            # Get the vertices of the Voronoi cell
            region_vertices = vor.vertices[region]
            
            # Compute polygon centroid using the shoelace formula
            if len(region_vertices) >= 3:
                x = region_vertices[:, 0]
                y = region_vertices[:, 1]
                # More accurate centroid using the formula
                A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                if abs(A) > 1e-12:
                    cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
                    cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
                    new_interior[i] = [cx, cy]
                else:
                    new_interior[i] = interior_points[i]
            else:
                new_interior[i] = interior_points[i]
        
        # Blend with old position for stability
        alpha = 0.5
        interior_points_new = alpha * new_interior + (1 - alpha) * interior_points
        
        # Check convergence
        if np.max(np.abs(interior_points_new - interior_points)) < 1e-8:
            interior_points = interior_points_new
            break
        
        interior_points = interior_points_new
        points[n_hull:] = interior_points
    
    # Verify the result
    def min_triangle_area(pts):
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
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 0.0
    
    a_min = min_triangle_area(points)
    a_hull = hull_area(points)
    
    if a_hull <= 0 or not np.all(np.isfinite(points)):
        # Fall back to a simple configuration
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points


# EVOLVE-BLOCK-END
