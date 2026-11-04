import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13
# Precompute indices for vectorization
_indices = np.array(list(itertools.combinations(range(N), 3)))

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area (Vectorized)
    p1 = points[_indices[:, 0]]
    p2 = points[_indices[:, 1]]
    p3 = points[_indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Generate diverse initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    inits.append(np.column_stack((x, y)).flatten())
    
    # 2. Jittered Regular 13-gon
    for _ in range(5):
        noise = np.random.normal(0, 0.1, size=(N, 2))
        base = np.column_stack((x, y))
        inits.append((base + noise).flatten())
        
    # 3. Random points in unit circle
    for _ in range(10):
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2 * np.pi, N)
        x_rand = r * np.cos(theta)
        y_rand = r * np.sin(theta)
        inits.append(np.column_stack((x_rand, y_rand)).flatten())
        
    # 4. Grid-like starts (spread out)
    # Try to place points on a grid within a square, then optimize
    # This is less critical than random/jittered but adds diversity
    
    best_points = None
    best_score = -float('inf')
    
    for x0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-8})
            
            opt_points = res.x.reshape(-1, 2)
            # Validate
            s = -_score(opt_points)
            
            if s > best_score:
                best_score = s
                best_points = opt_points
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((x, y))
        
    return best_points
