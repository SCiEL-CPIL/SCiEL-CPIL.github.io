import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    """
    # Remove duplicate points if any (shouldn't happen in our symmetric search)
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Calculate min triangle area
    min_area = float('inf')
    # 13 points -> 286 triangles
    # Vectorized calculation might be faster, but 286 is small
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def construct():
    best_score = 0.0
    best_points = None
    
    # Search parameters
    # r: ratio of inner radius to outer radius (outer is fixed at 1.0)
    # phi: angular offset of inner ring relative to outer ring (0 to 60 degrees)
    
    # Coarse search
    r_vals = np.linspace(0.1, 1.0, 50)
    phi_vals = np.linspace(0, np.pi/3, 36)
    
    # Outer points are fixed at radius 1, angles 0, 60, ...
    outer_angles = np.arange(6) * (np.pi / 3)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    for r in r_vals:
        for phi in phi_vals:
            inner_angles = outer_angles + phi
            inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            
            # Add center
            center = np.array([[0.0, 0.0]])
            
            points = np.vstack([center, outer_pts, inner_pts])
            
            # Check for duplicates (if r is very small or phi aligns with outer)
            # If r=0, inner points are all at center.
            # We should avoid r=0 to keep 13 distinct points if possible, 
            # but the problem says "13 points", usually implies distinct.
            # If duplicates exist, np.unique will reduce count, score might be invalid or 0.
            # Let's enforce r > 0.05 to be safe.
            
            score = _score_configuration(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    # Fine search around the best coarse solution
    if best_points is not None:
        # Extract best r and phi from best_points? 
        # Easier: just do a local grid search around the best (r, phi) found.
        # But we didn't store the params. Let's re-run a finer grid globally? 
        # 50x36 is fast. Let's do a second pass with higher resolution.
        
        r_vals_fine = np.linspace(0.1, 1.0, 200)
        phi_vals_fine = np.linspace(0, np.pi/3, 144)
        
        for r in r_vals_fine:
            for phi in phi_vals_fine:
                inner_angles = outer_angles + phi
                inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
                center = np.array([[0.0, 0.0]])
                points = np.vstack([center, outer_pts, inner_pts])
                
                score = _score_configuration(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

    return best_points
