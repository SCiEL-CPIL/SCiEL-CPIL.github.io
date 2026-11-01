import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def compute_score(points):
    # points: (N, 2)
    # Calculate min triangle area
    min_area = float('inf')
    # To speed up, we can just compute all
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    # Calculate hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, .volume is area
        if hull_area <= 1e-10:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Regular 14-gon
    # Vertices of regular 14-gon on unit circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Transform to fix P0 at (0,0) and P1 at (1,0)
    # P0 is pts[0], P1 is pts[1]
    # We want to apply a similarity transform (rotation + scaling + translation)
    # such that P0 -> (0,0) and P1 -> (1,0)
    
    p0 = pts[0]
    p1 = pts[1]
    v = p1 - p0
    length = np.linalg.norm(v)
    
    # Translation
    pts = pts - p0
    # Rotation
    # We want v to map to (length, 0) which then scales to (1,0)
    # Actually, just scale v to (1,0).
    # The angle of v is theta. Rotate by -theta.
    theta = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    pts = pts @ R.T
    # Scale
    pts = pts / length
    
    # Now pts[0] is (0,0) and pts[1] is (1,0)
    # Variables are pts[2:] flattened
    x0 = pts[2:].flatten()
    
    def objective(x):
        pts_var = np.zeros((N, 2))
        pts_var[0] = [0, 0]
        pts_var[1] = [1, 0]
        pts_var[2:] = x.reshape(-1, 2)
        
        # Optional: Check for degenerate hull
        if time.time() > deadline:
            return 0.0 # or keep current best? Minimize returns the value.
            
        score = compute_score(pts_var)
        return -score

    # Run Nelder-Mead
    best_x = x0
    best_val = -compute_score(pts)
    
    # We might run multiple restarts or just one long one.
    # Nelder-Mead on 24 dims can be slow.
    # Let's try one run with maxiter set based on time.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-6})
        if time.time() < deadline:
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
    except Exception:
        pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0, 0]
    final_pts[1] = [1, 0]
    final_pts[2:] = best_x.reshape(-1, 2)
    
    return final_pts
