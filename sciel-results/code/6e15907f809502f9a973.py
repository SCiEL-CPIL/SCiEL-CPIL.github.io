import numpy as np
from scipy.spatial import ConvexHull

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for duplicate points
    # If duplicates exist, min area is 0
    # We can check this by computing all pair distances, but that's O(N^2).
    # For N=13, it's fine.
    # Alternatively, if ConvexHull works, we can just compute areas.
    # If points are duplicates, triangle area is 0.
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Calculate min triangle area
    # Vectorized approach:
    # Create arrays for all combinations of 3 indices
    # i, j, k
    idx = np.arange(n)
    # We can use itertools.combinations to get indices, then vectorize the area calc
    # For N=13, 286 triangles. 
    # Let's stick to a loop for clarity and to avoid memory overhead of large arrays if N was bigger, 
    # but N is small.
    min_area = float('inf')
    
    # Pre-compute differences for speed?
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    
    for i in range(n):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            dx12 = p2[0] - p1[0]
            dy12 = p2[1] - p1[1]
            for k in range(j+1, n):
                p3 = points[k]
                dx13 = p3[0] - p1[0]
                dy13 = p3[1] - p1[1]
                area = 0.5 * abs(dx12 * dy13 - dx13 * dy12)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0 # Early exit if degenerate

    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def construct():
    best_score = 0.0
    best_points = None
    
    # Outer points: Regular Hexagon, Radius 1
    outer_angles = np.arange(6) * (np.pi / 3)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    center = np.array([[0.0, 0.0]])
    
    # Search space: r (inner radius), phi (inner rotation)
    # We search for r in [0.01, 0.99] and phi in [0.01, pi/3 - 0.01]
    # Avoiding 0 and pi/3 to prevent collinearity with outer points (which gives 0 area)
    # and to avoid duplicate configurations due to symmetry.
    
    # Coarse Grid
    r_vals = np.linspace(0.05, 0.95, 20)
    phi_vals = np.linspace(0.05, np.pi/3 - 0.05, 10)
    
    for r in r_vals:
        for phi in phi_vals:
            inner_angles = outer_angles + phi
            inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            points = np.vstack([center, outer_pts, inner_pts])
            
            score = _score_configuration(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    # Fine Grid around best coarse solution
    # To do this properly, we need to know the best r and phi.
    # Let's just do a global fine grid.
    # 20*10 = 200 evals.
    # Let's do 100*50 = 5000 evals for fine pass.
    
    r_vals_fine = np.linspace(0.05, 0.95, 100)
    phi_vals_fine = np.linspace(0.05, np.pi/3 - 0.05, 50)
    
    for r in r_vals_fine:
        for phi in phi_vals_fine:
            inner_angles = outer_angles + phi
            inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            points = np.vstack([center, outer_pts, inner_pts])
            
            score = _score_configuration(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    # Ensure we return a valid array
    if best_points is None:
        # Fallback: Return a regular 13-gon? Or the first point set?
        # A regular 13-gon is a valid convex set.
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
