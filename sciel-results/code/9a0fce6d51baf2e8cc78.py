import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3.0) / 2.0])

def area(p1, p2, p3):
    # 0.5 * |cross(p2-p1, p3-p1)|
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

def get_areas(points):
    # points: (N, 2)
    areas = np.zeros((N, N, N)) # Not efficient, just list
    areas_list = []
    for i, j, k in combinations(range(N), 3):
        areas_list.append(area(points[i], points[j], points[k]))
    return np.array(areas_list)

def min_triangle_area(points):
    return np.min(get_areas(points))

def soft_min(areas, beta):
    # Numerical stability
    min_a = np.min(areas)
    # If beta is large, exp(-beta * a) can underflow for a > min_a
    # Use log-sum-exp trick
    # log(sum(exp(-beta * (a - min_a)))) - beta * min_a ... 
    # Actually we want -1/beta * log(sum(exp(-beta * a)))
    # = -1/beta * (log(sum(exp(-beta * (a - min_a)))) - beta * min_a) ? No.
    # sum(exp(-beta * a)) = exp(-beta * min_a) * sum(exp(-beta * (a - min_a)))
    # log(...) = -beta * min_a + log(sum(exp(-beta * (a - min_a))))
    # -1/beta * log(...) = min_a - 1/beta * log(sum(exp(-beta * (a - min_a))))
    shifted = areas - min_a
    # shifted >= 0
    # exp(-beta * shifted) <= 1
    s = np.sum(np.exp(-beta * shifted))
    return min_a - (1.0 / beta) * np.log(s)

def penalty(points):
    # Barycentric coords check
    # p = v*V1 + w*V2 + u*V0
    # x = v + 0.5w
    # y = sqrt(3)/2 w
    # w = 2y/sqrt(3)
    # v = x - 0.5w
    # u = 1 - v - w
    pen = 0.0
    for p in points:
        w = 2.0 * p[1] / np.sqrt(3.0)
        v = p[0] - 0.5 * w
        u = 1.0 - v - w
        if u < 0: pen += u * u
        if v < 0: pen += v * v
        if w < 0: pen += w * w
    return pen

def cost_smooth(points_flat, beta):
    points = points_flat.reshape(N, 2)
    areas = get_areas(points)
    sm = soft_min(areas, beta)
    pen = penalty(points)
    return -sm + 10.0 * pen

def generate_inits():
    inits = []
    
    # 1. Random
    for _ in range(10):
        pts = []
        while len(pts) < N:
            # Rejection sampling in barycentric
            v = np.random.random()
            w = np.random.random() * (1.0 - v)
            x = v + 0.5 * w
            y = np.sqrt(3.0) / 2.0 * w
            pts.append([x, y])
        inits.append(np.array(pts))
        
    # 2. Grid in Barycentric
    # Try to distribute points evenly in the (v,w) triangle
    # 4x3 grid?
    # v in [0, 1], w in [0, 1-v]
    # Let's just use a few structured patterns
    
    # Pattern: Hexagonal lattice
    # Side length of triangle is 1.
    # Hexagonal grid spacing s.
    # For N=11, maybe s=0.33?
    # This is tricky to generate exactly.
    
    # Let's use a simple grid mapping
    # Map a square grid to the triangle using barycentric coords
    # n_x, n_y such that n_x * n_y >= 11
    # 4x3 = 12 points. Remove one? Or just take 11.
    for nx, ny in [(4, 3), (3, 4)]:
        pts = []
        for i in range(nx):
            for j in range(ny):
                if len(pts) >= N: break
                # i, j in [0, 1]
                # u = 1 - i/nx - j/ny ... no, need to map to triangle
                # Use: v = i/(nx-1) * (1 - j/(ny-1)) ? No.
                # Standard mapping:
                # v = i / (nx - 1)
                # w = j / (ny - 1) * (1 - v)
                if nx > 1:
                    v = i / (nx - 1)
                else:
                    v = 0.5
                if ny > 1:
                    w = j / (ny - 1) * (1.0 - v)
                else:
                    w = 0.5 * (1.0 - v)
                x = v + 0.5 * w
                y = np.sqrt(3.0) / 2.0 * w
                pts.append([x, y])
        if len(pts) == N:
             inits.append(np.array(pts))
             
    # 3. Symmetric "Star" or "Ring"
    # Center + 10 around?
    c = [0.5, np.sqrt(3)/6] # Centroid
    pts = [c]
    # 5 points at radius r1, 5 at r2?
    # Or 10 points on a hexagon?
    # Let's just rely on random and grid.
    
    return inits

def construct():
    start_time = time.time()
    timeout = 145.0 # 5s margin
    
    best_points = None
    best_score = -1.0
    
    inits = generate_inits()
    
    # If no inits generated, fallback to random
    if not inits:
        pts = []
        while len(pts) < N:
            v = np.random.random()
            w = np.random.random() * (1.0 - v)
            x = v + 0.5 * w
            y = np.sqrt(3.0) / 2.0 * w
            pts.append([x, y])
        inits = [np.array(pts)]
        
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        current_pts = init.copy()
        current_score = min_triangle_area(current_pts)
        
        # Annealing on beta
        betas = [5.0, 20.0, 100.0, 500.0]
        
        for beta in betas:
            if time.time() - start_time > timeout:
                break
            
            # Optimize using Nelder-Mead
            # Flatten points
            x0 = current_pts.flatten()
            
            # Define objective with current beta
            def obj(x):
                return cost_smooth(x, beta)
                
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
                if res.success:
                    current_pts = res.x.reshape(N, 2)
                    # Update score
                    score = min_triangle_area(current_pts)
                    if score > current_score:
                        current_score = score
            except Exception:
                pass
        
        # Final polish with high beta
        if time.time() - start_time < timeout:
             # Use very high beta, essentially raw min
             # Nelder-Mead on raw min area
             def obj_raw(x):
                 pts = x.reshape(N, 2)
                 ma = min_triangle_area(pts)
                 pen = penalty(pts)
                 return -ma + 10.0 * pen
                 
             x0 = current_pts.flatten()
             try:
                 res = minimize(obj_raw, x0, method='Nelder-Mead', 
                                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9})
                 if res.success:
                     current_pts = res.x.reshape(N, 2)
                     current_score = min_triangle_area(current_pts)
             except Exception:
                 pass
                 
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts
            
    if best_points is None:
        best_points = inits[0]
        
    return best_points
