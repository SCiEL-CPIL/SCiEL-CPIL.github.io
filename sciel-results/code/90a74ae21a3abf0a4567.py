import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

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

def negative_score(points_flat):
    points = points_flat.reshape(N, 2)
    s = score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initial configuration: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0  # Will be normalized by hull area in score, so scale doesn't matter for ratio
    x0 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]).flatten()
    
    best_x = x0.copy()
    best_score = score(best_x.reshape(N, 2))
    
    # Try to optimize
    try:
        res = minimize(negative_score, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'fatol': 1e-8, 'xatol': 1e-8})
        
        if time.time() < deadline:
            candidate = res.x.reshape(N, 2)
            cand_score = score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = res.x.copy()
    except Exception as e:
        pass
    
    # If time is running out, return best found
    if time.time() > deadline:
        return best_x.reshape(N, 2)
    
    return best_x.reshape(N, 2)
