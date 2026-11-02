import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Function to convert barycentric coords (u, v) to Cartesian
    # P = u*v0 + v*v1 + (1-u-v)*v2
    # Since v0 is (0,0), P = v*v1 + (1-u-v)*v2
    def bary_to_cart(u, v):
        w = 1.0 - u - v
        return v * v1 + w * v2

    # Function to calculate min area for a set of points
    def calc_min_area(points):
        # points is an array of shape (N, 2)
        n = len(points)
        min_area = float('inf')
        # Vectorized calculation of areas for all triples
        # This might be memory heavy if done all at once, so loop over i
        for i in range(n-2):
            for j in range(i+1, n-1):
                # Vectorize over k
                p_k = points[j+1:]
                p_i = points[i]
                p_j = points[j]
                
                # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
                # cross product in 2D: (x1*y2 - y1*x2)
                v1 = p_j - p_i
                v2 = p_k - p_i
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                min_area = min(min_area, np.min(areas))
        return min_area

    def objective_strategy1(x):
        # x is flat array of 16 barycentric coords for 8 interior points
        # Fixed points: v0, v1, v2
        # Interior points: 8
        pts = []
        pts.append(v0)
        pts.append(v1)
        pts.append(v2)
        
        for i in range(8):
            u = x[2*i]
            v = x[2*i+1]
            # Check bounds, though DE should handle it
            if u < 0: u = 0
            if v < 0: v = 0
            if u + v > 1: 
                # Scale to boundary
                s = 1.0 / (u+v)
                u *= s
                v *= s
            p = bary_to_cart(u, v)
            pts.append(p)
            
        pts = np.array(pts)
        min_area = calc_min_area(pts)
        return -min_area # Minimize negative area

    def objective_strategy2(x):
        # x is flat array of 22 barycentric coords for 11 points
        pts = []
        for i in range(11):
            u = x[2*i]
            v = x[2*i+1]
            if u < 0: u = 0
            if v < 0: v = 0
            if u + v > 1:
                s = 1.0 / (u+v)
                u *= s
                v *= s
            p = bary_to_cart(u, v)
            pts.append(p)
        pts = np.array(pts)
        min_area = calc_min_area(pts)
        return -min_area

    best_points = None
    best_score = 0.0

    # Strategy 1: Fixed vertices
    if time.time() < deadline - 20:
        bounds1 = [(0, 1)] * 16
        try:
            res1 = differential_evolution(objective_strategy1, bounds1, seed=42, maxiter=50, popsize=10, polish=False)
            if res1.fun < 0:
                score1 = -res1.fun
                if score1 > best_score:
                    best_score = score1
                    # Reconstruct points
                    x = res1.x
                    pts = [v0, v1, v2]
                    for i in range(8):
                        u = x[2*i]
                        v = x[2*i+1]
                        if u+v > 1:
                            s = 1.0/(u+v)
                            u*=s; v*=s
                        pts.append(bary_to_cart(u, v))
                    best_points = np.array(pts)
        except Exception as e:
            pass

    # Strategy 2: Free points
    if time.time() < deadline - 10:
        bounds2 = [(0, 1)] * 22
        try:
            res2 = differential_evolution(objective_strategy2, bounds2, seed=42, maxiter=30, popsize=10, polish=False)
            if res2.fun < 0:
                score2 = -res2.fun
                if score2 > best_score:
                    best_score = score2
                    x = res2.x
                    pts = []
                    for i in range(11):
                        u = x[2*i]
                        v = x[2*i+1]
                        if u+v > 1:
                            s = 1.0/(u+v)
                            u*=s; v*=s
                        pts.append(bary_to_cart(u, v))
                    best_points = np.array(pts)
        except Exception as e:
            pass

    # Fallback if no optimization succeeded
    if best_points is None:
        # Simple random or grid
        rng = np.random.RandomState(42)
        pts = []
        # Vertices
        pts.append(v0)
        pts.append(v1)
        pts.append(v2)
        # Centroid
        pts.append((v0+v1+v2)/3)
        # Random interior
        for _ in range(7):
            u = rng.random()
            v = rng.random()
            if u+v > 1:
                u, v = 1-u, 1-v
            pts.append(bary_to_cart(u, v))
        best_points = np.array(pts)

    return best_points
