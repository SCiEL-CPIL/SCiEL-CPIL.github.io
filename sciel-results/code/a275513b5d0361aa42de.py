# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    # Iterate over all combinations of 3 points
    # For n=13, C(13,3) = 286, which is very fast.
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Calculate the score A_min / A_hull."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
        a_min = _min_triangle_area(points)
        return a_min / a_hull
    except Exception:
        return 0.0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area 
    relative to the convex hull area.
    
    This implementation uses a two-ring configuration and searches over 
    different splits (e.g., 7 outer / 6 inner, 6 outer / 7 inner, etc.), 
    inner radii, and rotational offsets to maximize the score.
    """
    best_score = -1
    best_points = None
    
    # Define candidate splits (n_outer, n_inner)
    splits = [(7, 6), (6, 7), (5, 8), (8, 5), (4, 9), (9, 4)]
    
    # Search space for inner radius and rotation angle
    radii = np.linspace(0.2, 0.9, 30)
    # We search rotations in the range of one vertex spacing of the outer polygon
    # to cover unique symmetries
    for n_outer, n_inner in splits:
        # Precompute outer points
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        x_outer = np.cos(angles_outer)
        y_outer = np.sin(angles_outer)
        points_outer = np.column_stack((x_outer, y_outer))
        
        # Precompute base inner points (radius 1, angle 0)
        angles_inner_base = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        x_inner_base = np.cos(angles_inner_base)
        y_inner_base = np.sin(angles_inner_base)
        points_inner_base = np.column_stack((x_inner_base, y_inner_base))
        
        # Search over radii and rotations
        for r in radii:
            # Scale inner points
            points_inner_scaled = points_inner_base * r
            
            # Search rotations
            # Step size for rotation
            rot_steps = 20
            for i in range(rot_steps):
                rot = (i / rot_steps) * (2 * np.pi / n_outer)
                cos_rot = np.cos(rot)
                sin_rot = np.sin(rot)
                rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
                points_inner_rot = points_inner_scaled @ rot_matrix.T
                
                # Combine points
                points = np.vstack([points_outer, points_inner_rot])
                
                # Calculate score
                score = _score(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
    # Fallback to regular 13-gon if something goes wrong
    if best_points is None or best_score < 0:
         n = 13
         angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
         x = np.cos(angles)
         y = np.sin(angles)
         best_points = np.column_stack((x, y))

    return best_points

# EVOLVE-BLOCK-END
