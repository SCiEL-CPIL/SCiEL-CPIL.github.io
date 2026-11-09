import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(points):
    n = points.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return -1e10
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _param_to_points(params):
    # params: [r0, ..., r13, dtheta0, ..., dtheta13]
    # r in [0.8, 1.2], dtheta in [-pi/14, pi/14]
    r = params[:N]
    dtheta = params[N:]
    
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = base_angles + dtheta
    
    # Ensure points are in counter-clockwise order by sorting angles
    # This maintains convexity if r > 0 and angle gaps are sufficient
    # For safety, we just compute points. The objective will penalize non-convex or small areas.
    # Actually, to strictly ensure convex hull is the 14-gon, we should ensure no point is inside.
    # With r ~ 1 and small dtheta, it's almost certainly convex.
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack([x, y])

def _neg_objective(params):
    points = _param_to_points(params)
    
    # Check for degeneracy
    if np.any(np.abs(points) < 1e-5):
        return 1e10
        
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area(points)
    if min_area < 1e-10:
        return 1e10
        
    # We want to maximize min_area / hull_area
    # Equivalent to maximizing log(min_area) - log(hull_area)
    # Minimize negative of that
    return -(np.log(min_area) - np.log(hull_area))

def _neg_objective_smooth(params):
    # Add a small penalty for non-uniformity to help escape flat regions?
    # Or just use the same one. L-BFGS-B is robust enough.
    return _neg_objective(params)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Bounds for parameters
    # r: [0.5, 1.5] (allowing some flexibility)
    # dtheta: [-0.5, 0.5] (radians, ~30 degrees, plenty of room)
    bounds = [(0.5, 1.5)] * N + [(-0.5, 0.5)] * N
    
    # Initial guesses
    # 1. Regular polygon (r=1, dtheta=0)
    x0_regular = np.zeros(2*N)
    x0_regular[:N] = 1.0
    x0_regular[N:] = 0.0
    
    # 2. Perturbed regular polygon
    rng = np.random.default_rng(42)
    guesses = [x0_regular]
    
    # Generate diverse starting points
    for i in range(20):
        x0 = np.zeros(2*N)
        # Random radii close to 1
        x0[:N] = 1.0 + rng.normal(0, 0.1, N)
        # Random angles close to regular
        x0[N:] = rng.normal(0, 0.05, N)
        guesses.append(x0)
        
        # Some more extreme perturbations
        if i % 5 == 0:
            x0[:N] = 1.0 + rng.normal(0, 0.3, N)
            x0[N:] = rng.normal(0, 0.1, N)
            guesses.append(x0)

    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_neg_objective, guess, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline:
                break
                
            score = -res.fun
            # The score from objective is log(min_area/hull_area)
            # We need to convert back to min_area/hull_area for comparison with standard score
            # Actually, let's just compute the standard score for the resulting points
            pts = _param_to_points(res.x)
            std_score = _score(pts)
            
            if std_score > best_score:
                best_score = std_score
                best_params = res.x
                best_points = pts
        except Exception:
            pass

    # If no points found, return regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
