import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 11

def construct():
    # Vertices of the equilateral triangle
    # A = (0, 0), B = (1, 0), C = (0.5, sqrt(3)/2)
    
    def is_inside(pt):
        x, y = pt
        if y < -1e-9: return False
        if y > np.sqrt(3) * x + 1e-9: return False
        if y > -np.sqrt(3) * (x - 1) + 1e-9: return False
        return True

    def min_area(points):
        min_a = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < min_a:
                min_a = area
        return min_a

    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalties for being outside
        pen1 = np.sum(np.maximum(0, -y)**2)
        pen2 = np.sum(np.maximum(0, y - np.sqrt(3)*x)**2)
        pen3 = np.sum(np.maximum(0, y + np.sqrt(3)*(x-1))**2)
        
        penalty = 1000 * (pen1 + pen2 + pen3)
        
        min_a = min_area(pts)
        
        return -min_a + penalty

    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N

    # Run Differential Evolution
    # Using multiple seeds to increase chance of finding global optimum
    best_points = None
    best_score = -float('inf')
    
    for seed in [42, 123, 999]:
        try:
            res = differential_evolution(objective, bounds, maxiter=100, popsize=15, tol=1e-5, seed=seed, polish=True)
            if res.fun < 100: # Check if it found a valid solution (low penalty)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        np.random.seed(42)
        random_pts = []
        while len(random_pts) < N:
            x = np.random.rand()
            y = np.random.rand() * np.sqrt(3)/2
            if is_inside([x, y]):
                random_pts.append([x, y])
        best_points = np.array(random_pts)

    return best_points
