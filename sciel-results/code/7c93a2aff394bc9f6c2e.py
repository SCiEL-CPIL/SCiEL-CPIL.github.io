# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize the 
    ratio of the smallest triangle area to the convex hull area.
    """
    n = 13
    
    # Helper function to calculate the metric
    def calculate_metric(points):
        try:
            # Calculate areas of all triangles
            # n=13, C(13,3) = 286 triangles
            min_area = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                        area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                        if area < min_area:
                            min_area = area
            
            # Calculate Convex Hull Area
            hull = ConvexHull(points)
            hull_area = hull.volume # In 2D, volume is area
            
            if hull_area <= 0:
                return 0.0
                
            return min_area / hull_area
        except Exception:
            return 0.0

    def objective(points_flat):
        points = points_flat.reshape(n, 2)
        # Normalize to prevent scaling issues, keep in unit box roughly
        # Center at 0
        points = points - np.mean(points, axis=0)
        # Scale so that max coordinate is 1 (or similar) to keep numbers stable
        max_coord = np.max(np.abs(points))
        if max_coord > 0:
            points = points / max_coord
        return -calculate_metric(points)

    # Initial Guess: Hexagonal Lattice
    # Center at (0,0)
    # Ring 1: 6 points at distance 1
    # Ring 2: 6 points at distance 2 (offset)
    # This forms a hexagon of radius 2 with 13 points.
    
    d = 1.0
    angles = np.deg2rad(np.arange(0, 360, 60))
    
    points = np.array([[0.0, 0.0]])
    
    # Ring 1
    for a in angles:
        x = d * np.cos(a)
        y = d * np.sin(a)
        points = np.vstack([points, [x, y]])
        
    # Ring 2
    # The outer ring of a hexagonal grid of radius 2 has 12 points? No.
    # Radius 0: 1 point
    # Radius 1: 6 points
    # Radius 2: 12 points
    # Total: 1 + 6 + 12 = 19 points. Too many.
    
    # We need 13 points.
    # 1 (center) + 6 (ring 1) + 6 (subset of ring 2).
    # Let's take the 6 points from ring 2 that are aligned with the ring 1 points.
    # This creates a "star" shape or a stretched hexagon.
    # Alternatively, a regular hexagon (6 vertices) + 6 midpoints of edges + 1 center = 13.
    # This is a very good candidate.
    
    # Regular Hexagon vertices
    r = 1.0
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.array([[r * np.cos(a), r * np.sin(a)] for a in hex_angles])
    
    # Midpoints of edges
    mids = []
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        mid = (p1 + p2) / 2.0
        mids.append(mid)
    mids = np.array(mids)
    
    # Center
    center = np.array([[0.0, 0.0]])
    
    # Combine: 6 vertices + 6 midpoints + 1 center = 13 points
    initial_points = np.vstack([hex_verts, mids, center])
    
    # Flatten for optimizer
    initial_flat = initial_points.flatten()
    
    # Optimization
    best_points = initial_points
    best_score = calculate_metric(best_points)
    
    start_time = time.time()
    time_limit = 600 # seconds
    
    # Try a few optimization methods or iterations
    # Nelder-Mead is robust for non-smooth functions
    try:
        from scipy.optimize import minimize
        
        # We might need multiple restarts or a longer run
        # Since the function is non-smooth, Nelder-Mead is a good choice
        # But 26 dimensions is high for NM. Let's try it with a reasonable maxiter.
        
        # To speed up, we can use a simpler local search or just return the heuristic if time is tight.
        # The heuristic (Hexagon + Midpoints + Center) is already a strong candidate.
        # Let's try to polish it.
        
        # We will run minimize for a limited time
        # Since we can't easily interrupt minimize, we will run it with a maxiter that fits in time.
        # Estimating 1000 iterations for NM in 26D might take a few seconds.
        
        result = minimize(objective, initial_flat, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < -best_score:
            best_points = result.x.reshape(n, 2)
            best_score = -result.fun
            
    except Exception as e:
        pass # Keep initial points if optimization fails
    
    # Check time and potentially do a second pass with a different initial guess if time allows
    # But the hexagonal symmetry is likely near-optimal for this n.
    
    # Normalize the final points to be in a standard position (e.g., centered at 0,0)
    # The metric is scale invariant, so position doesn't matter for the score, 
    # but we must return finite coordinates.
    
    # Ensure no NaNs or Infs
    if not np.all(np.isfinite(best_points)):
         # Fallback to initial points
         best_points = initial_points
         
    return best_points

# EVOLVE-BLOCK-END
