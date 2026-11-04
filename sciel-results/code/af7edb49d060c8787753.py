import numpy as np
from scipy.optimize import minimize
import time

N = 14

def triangle_areas(points):
    # points: (N, 2)
    # returns array of areas for all triples
    n = len(points)
    areas = np.zeros((n, n, n)) # or just compute min directly?
    # Vectorized computation of all triangle areas
    # p_i, p_j, p_k
    # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    
    # To save memory/time, we can compute in chunks or just use loops if N=14 is small.
    # 14 choose 3 = 364 triangles. This is tiny.
    # Looping 364 times is negligible.
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product magnitude
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def convex_hull_area(points):
    # Simple convex hull area calculation
    # Sort by angle? Or use monotone chain?
    # For N=14, even O(N^3) hull check is fine, but O(N log N) is easy.
    # Let's use scipy.spatial if available, but manual is safer for speed/dependency.
    # Monotone chain algorithm.
    
    pts = np.array(points)
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
        
    # Shoelace formula
    x = np.array([h[0] for h in hull])
    y = np.array([h[1] for h in hull])
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def objective(x):
    points = x.reshape(-1, 2)
    min_area = triangle_areas(points)
    hull_area = convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    score = min_area / hull_area
    return -score

def construct():
    deadline = time.time() + 110
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Init: Regular polygon with noise
        angles = 2 * np.pi * np.arange(N) / N
        radii = 1.0 + 0.1 * rng.normal(0, 1, N)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        x0 = points.flatten()
        
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_score: # Note: objective returns -score, so we want min fun
            best_score = res.fun
            best_points = res.x.reshape(-1, 2)
            
    if best_points is None:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
