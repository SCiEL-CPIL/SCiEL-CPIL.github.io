import numpy as np
from scipy.optimize import differential_evolution

N = 11
H = np.sqrt(3) / 2

def inside_triangle(pts):
    # pts: (N, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    # y >= 0
    if np.any(y < -1e-9):
        return False
    # y <= sqrt(3) x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    # y <= -sqrt(3) (x - 1) => y + sqrt(3)x <= sqrt(3)
    if np.any(y + np.sqrt(3) * x > np.sqrt(3) + 1e-9):
        return False
    return True

def min_triangle_area(pts):
    # Compute min area of all triangles
    n = len(pts)
    min_area = np.inf
    # Vectorized computation might be tricky for all triples, 
    # but N=11 is small.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    pts = x.reshape(-1, 2)
    if not inside_triangle(pts):
        # Penalty: return a large positive value (since we minimize)
        # The max possible area is area of triangle / 2? No, max area is area of container.
        # Area of container = 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4 ~ 0.433
        # So a penalty of 10 is sufficient to be "bad".
        return 10.0
    area = min_triangle_area(pts)
    return -area

def construct():
    # Bounds for differential evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1)] * N + [(0, H)] * N
    
    # Use differential evolution
    # seed for reproducibility
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=200, 
        tol=1e-6,
        popsize=20,
        polish=True
    )
    
    best_pts = res.x.reshape(-1, 2)
    
    # Ensure valid
    if not inside_triangle(best_pts):
        # Fallback to random if optimization failed
        rng = np.random.default_rng(42)
        while True:
            pts = rng.random((N, 2))
            pts[:, 0] *= 1.0
            pts[:, 1] *= H
            if inside_triangle(pts):
                break
        best_pts = pts
        
    return best_pts
