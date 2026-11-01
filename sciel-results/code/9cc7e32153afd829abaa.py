import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def compute_score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    # Check for degeneracy (collinear points, etc.)
    # ConvexHull requires at least 3 non-collinear points in 2D
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # Calculate areas of all triangles
    # There are N choose 3 triangles
    # For efficiency, we can vectorize
    # But for N=14, 364 triangles is small.
    min_area = float('inf')
    
    # Get indices
    indices = range(N)
    for i, j, k in itertools.combinations(indices, 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Area = 0.5 * |det(p2-p1, p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        if area < min_area:
            min_area = area
            if min_area < 1e-12:
                return 0.0

    return min_area / hull_area

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Fix gauge: p0 at (0,0), p1 at (1,0)
    # This removes translation and rotation invariance.
    # We optimize the remaining N-2 points.
    
    # Transform points_init to gauge
    p0 = points_init[0]
    p1 = points_init[1]
    
    # Vector from p0 to p1
    v = p1 - p0
    dist = np.linalg.norm(v)
    
    # Rotation matrix to align v with x-axis
    # We want to map v to (dist, 0)
    # Then scale to make it (1,0)
    
    # Actually, simpler:
    # Translate p0 to 0
    pts_trans = points_init - p0
    # Rotate so p1 is on x-axis
    # p1_trans = pts_trans[1]
    angle_to_rotate = -np.arctan2(pts_trans[1, 1], pts_trans[1, 0])
    cos_a = np.cos(angle_to_rotate)
    sin_a = np.sin(angle_to_rotate)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts_rot = pts_trans @ rot_mat.T
    
    # Scale so p1 is at (1,0)
    # p1_rot should be (dist, 0)
    scale = 1.0 / pts_rot[1, 0]
    pts_scaled = pts_rot * scale
    
    # Now pts_scaled[0] is (0,0) and pts_scaled[1] is (1,0)
    # We optimize pts_scaled[2:]
    
    x0 = pts_scaled[2:].flatten()
    
    # Objective function
    def objective(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(N-2, 2)
        
        # Check for very small areas or invalid hull
        # If points are too close, score is 0
        score = compute_score(pts)
        return -score # Minimize negative score to maximize score
    
    # Optimization
    best_x = x0
    best_score = compute_score(pts_scaled)
    
    # Use Powell method, which is derivative-free and good for this
    # We need to be careful with time
    # Let's run multiple restarts or just one long run?
    # One long run with Powell is usually good.
    
    try:
        res = minimize(objective, x0, method='Powell', 
                       options={'maxiter': 1000, 'disp': False})
        
        if res.fun < -best_score: # res.fun is -score
            best_x = res.x
            best_score = -res.fun
            
    except Exception as e:
        pass
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5.0:
         # If we have time, maybe try a different initial guess?
         # Or refine with Nelder-Mead?
         pass

    # Reconstruct best points
    best_pts = np.zeros((N, 2))
    best_pts[0] = [0.0, 0.0]
    best_pts[1] = [1.0, 0.0]
    best_pts[2:] = best_x.reshape(N-2, 2)
    
    return best_pts
