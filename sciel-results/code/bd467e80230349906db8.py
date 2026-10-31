import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = compute_min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: 3x3 grid + 4 points
    # 3x3 grid in [0, 1]x[0, 1]
    grid = np.array([[i/2.0, j/2.0] for i in range(3) for j in range(3)])
    
    # Add 4 points. Start with random points in a slightly larger box to allow expansion
    rng = np.random.RandomState(42)
    extra = rng.uniform(-0.5, 1.5, size=(4, 2))
    
    points = np.vstack([grid, extra])
    
    best_score = score(points)
    best_points = points.copy()
    
    # Local optimization
    # We optimize the 26 coordinates
    from scipy.optimize import minimize
    
    def objective(x):
        p = x.reshape(13, 2)
        return -score(p)
        
    # Use Nelder-Mead
    options = {
        'maxiter': 1000,
        'xatol': 1e-4,
        'fatol': 1e-6
    }
    
    # We might need multiple restarts or a better initial guess
    # Try optimizing the current guess
    try:
        res = minimize(objective, points.flatten(), method='Nelder-Mead', options=options)
        if res.fun < -best_score:
            best_score = -res.fun
            best_points = res.x.reshape(13, 2)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_points
        
    # Try a different initial guess: 4x4 grid minus 3 points
    # 4x4 grid in [0, 1]x[0, 1]
    grid4 = np.array([[i/3.0, j/3.0] for i in range(4) for j in range(4)])
    # Remove 3 corners
    indices_to_keep = [i for i in range(16) if i not in [0, 3, 12]] # Remove (0,0), (3,0), (0,3)
    points2 = grid4[indices_to_keep]
    
    score2 = score(points2)
    if score2 > best_score:
        best_score = score2
        best_points = points2.copy()
        
    # Optimize this one too if time permits
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, points2.flatten(), method='Nelder-Mead', options={'maxiter': 500})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_points = res2.x.reshape(13, 2)
        except Exception:
            pass
            
    return best_points
