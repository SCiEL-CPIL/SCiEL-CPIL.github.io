import numpy as np
from scipy.optimize import minimize
import itertools
import time

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

    # Pre-compute indices for speed
    indices = list(itertools.combinations(range(N), 3))

    def objective_smooth(flat_pts, beta=20.0):
        pts = flat_pts.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalties for being outside
        pen1 = np.sum(np.maximum(0, -y)**2)
        pen2 = np.sum(np.maximum(0, y - np.sqrt(3)*x)**2)
        pen3 = np.sum(np.maximum(0, y + np.sqrt(3)*(x-1))**2)
        penalty = 100 * (pen1 + pen2 + pen3)
        
        if penalty > 1e-3:
            return 1000.0 + penalty
            
        # Calculate areas
        areas = []
        for i, j, k in indices:
            p1, p2, p3 = pts[i], pts[j], pts[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            areas.append(area)
            
        areas = np.array(areas)
        
        # Log-Sum-Exp approximation of min
        # We want to minimize -min(areas)
        # -min(areas) approx 1/beta * log(sum(exp(-beta * areas)))
        # Stable computation:
        neg_areas = -areas
        max_neg = np.max(neg_areas)
        lse = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        
        return lse / beta + penalty

    # Multi-start initialization
    best_score = -float('inf')
    best_points = None
    
    starts = []
    
    # Start 1: Random
    np.random.seed(42)
    random_pts = []
    while len(random_pts) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            random_pts.append([x, y])
    starts.append(np.array(random_pts).flatten())
    
    # Start 2: Grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + 0.2*i
            y = 0.1 + 0.2*j * 0.866
            if is_inside([x, y]):
                grid_pts.append([x, y])
    while len(grid_pts) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            grid_pts.append([x, y])
    starts.append(np.array(grid_pts[:N]).flatten())
    
    # Start 3: Symmetric-ish
    sym_pts = [
        [0.5, 0.288],
        [0.25, 0.144], [0.75, 0.144], [0.5, 0.432],
        [0.1, 0.3], [0.9, 0.3], [0.5, 0.7],
        [0.05, 0.05], [0.95, 0.05], [0.2, 0.6], [0.8, 0.6]
    ]
    valid_sym = []
    for p in sym_pts:
        if is_inside(p):
            valid_sym.append(p)
    while len(valid_sym) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            valid_sym.append([x, y])
    starts.append(np.array(valid_sym[:N]).flatten())

    start_time = time.time()
    
    for start in starts:
        if time.time() - start_time > 100:
            break
        try:
            # Optimize the smoothed objective
            res = minimize(lambda x: objective_smooth(x, beta=20.0), start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1000:
                pts = res.x.reshape(N, 2)
                valid = True
                for p in pts:
                    if not is_inside(p):
                        valid = False
                        break
                if valid:
                    score = min_area(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass
            
    if best_points is None:
        return np.array(random_pts)
        
    return best_points
