import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_score(points):
    N = len(points)
    if N < 3:
        return 0.0
    
    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0

    # Vectorized triangle area calculation
    # For each pair (i, j), compute area with all k
    # Area = 0.5 * | cross(p_j - p_i, p_k - p_i) |
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # cross = x_j y_k - y_j x_k - x_i y_k + y_i x_k + x_i y_j - y_i x_j
    
    # Let's just do the loops but with numpy operations inside to be safe and fast enough for 13 points
    # 13C3 = 286 triangles.
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Strategy: 12-gon + center
    # 12 points on a circle (or near circle), 1 point at center (or near center)
    # This topology allows for larger min triangle areas than a regular 13-gon
    
    # Initial guess: 12 points on unit circle, 1 at origin
    # Angles for 12 points: 0, 30, 60, ..., 330 degrees
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points_init = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    center = np.array([0.0, 0.0])
    points_init = np.vstack([points_init, center])
    
    s = get_score(points_init)
    best_score = s
    best_points = points_init.copy()
    
    # Define objective function for optimization
    # Flatten points to 2D array of size 26
    def neg_score_flat(x_flat):
        pts = x_flat.reshape(-1, 2)
        return -get_score(pts)
    
    # Optimization loop
    # We will use Nelder-Mead on the 2D coordinates
    # Since the problem is somewhat symmetric, we can try to break symmetry with random restarts
    
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Determine starting point for this restart
        if n_restarts == 1:
            x_start = best_points.flatten()
        else:
            # Perturb the best solution found so far
            # Or create a new random 12-gon + center
            if n_restarts % 5 == 0:
                # New random 12-gon + center
                # Randomize angles slightly
                angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.random.normal(0, 0.2, 12)
                pts = np.column_stack((np.cos(angles), np.sin(angles)))
                c = np.random.normal(0, 0.1, 2)
                x_start = np.vstack([pts, c]).flatten()
            else:
                # Perturb best
                x_start = best_points.flatten() + np.random.normal(0, 0.1, 26)
        
        try:
            res = minimize(neg_score_flat, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True})
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(-1, 2)
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    return best_points
