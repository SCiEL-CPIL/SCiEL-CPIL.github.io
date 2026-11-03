import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

def _area(p1, p2, p3):
    # p1, p2, p3 are (2,) arrays
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _all_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_area(points[i], points[j], points[k]))
    return np.array(areas)

def _project(points):
    # Project points into the triangle
    pts = np.array(points, dtype=float).copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Bottom
        if y < 0:
            y = 0
        # Left: y <= sqrt(3)*x  => sqrt(3)*x - y >= 0
        if H * 2 * x - y < 0: # sqrt(3)*x - y < 0
            # Project
            x_new = (x + H * 2 * y) / 4 # (x + sqrt(3)y)/4
            y_new = (3 * y + H * 2 * x) / 4 # (3y + sqrt(3)x)/4
            x, y = x_new, y_new
        # Right: y <= sqrt(3)*(1-x) => sqrt(3)*x + y - sqrt(3) <= 0
        if H * 2 * x + y - H * 2 > 0:
            # Project
            x_new = (x - H * 2 * y + 3) / 4 # (x - sqrt(3)y + 3)/4
            y_new = (3 * y - H * 2 * x + H * 2) / 4 # (3y - sqrt(3)x + sqrt(3))/4
            x, y = x_new, y_new
        # Re-check bottom after projections
        if y < 0:
            y = 0
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_val = -1
    best_points = None
    
    # Initial random points
    rng = np.random.default_rng(42)
    
    # Number of restarts to attempt
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        # Use a grid-like or random initialization
        # Random inside triangle
        pts = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = 0.5 * u + 0.5 * v
                y = H * (1 - u - v) # Wait, standard transform
                # Let's use standard barycentric
                # P = u*A + v*B + w*C, u+v+w=1
                # A=(0,0), B=(1,0), C=(0.5, H)
                # x = v + 0.5w, y = H*w
                # u = 1-v-w
                # v, w >= 0, v+w <= 1
                # x = v + 0.5(1-u-v)? No.
                # Let's just use rejection sampling for init
                while True:
                    x, y = rng.random(), rng.random()
                    if y <= H * x and y <= H * (1 - x):
                        break
            else:
                # retry
                continue
            pts.append([x, y])
        
        if len(pts) < N:
             # Fallback to random
             pts = []
             for _ in range(N):
                 while True:
                     x, y = rng.random(), rng.random()
                     if y <= H * x and y <= H * (1 - x):
                         break
                 pts.append([x, y])
        
        pts = np.array(pts)
        pts = _project(pts)
        
        # Define objective
        def objective(p):
            p = p.reshape(N, 2)
            p = _project(p)
            areas = _all_areas(p)
            # Soft min
            beta = 100.0
            min_area = -np.log(np.sum(np.exp(-beta * areas))) / beta
            # Return negative to maximize
            return -min_area
        
        # Optimize
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() > deadline:
                break
                
            opt_points = _project(res.x.reshape(N, 2))
            val = -res.fun
            # Calculate true min area for this result
            true_areas = _all_areas(opt_points)
            true_min = np.min(true_areas)
            
            if true_min > best_val:
                best_val = true_min
                best_points = opt_points
                
        except Exception:
            continue
            
    if best_points is None:
        # Return some valid points
        pts = []
        for _ in range(N):
            while True:
                x, y = rng.random(), rng.random()
                if y <= H * x and y <= H * (1 - x):
                    break
            pts.append([x, y])
        return np.array(pts)
        
    return best_points
