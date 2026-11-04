import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0

def min_triangle_area(points):
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                    (points[k][0] - points[i][0]) * (points[j][1] - points[i][1])
                )
                if area < min_a:
                    min_a = area
    return min_a

def is_inside(points, eps=1e-9):
    for p in points:
        x, y = p
        if y < -eps or y > S3 * x + eps or y > S3 * (1 - x) + eps:
            return False
    return True

def objective(flat_points):
    points = flat_points.reshape(N, 2)
    pen = 0.0
    for p in points:
        x, y = p
        if y < 0:
            pen += abs(y)
        if y > S3 * x:
            pen += (y - S3 * x)
        if y > S3 * (1 - x):
            pen += (y - S3 * (1 - x))
    
    if pen > 1e-6:
        return 100.0 + pen
    
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    bounds = [(0, 1), (0, H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=30, 
            popsize=15,
            tol=1e-3,
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.7,
            init='latinhypercube'
        )
        
        best_points = res.x.reshape(N, 2)
        best_score = -res.fun
        
        if best_points is not None and is_inside(best_points):
            pass
        else:
            best_points = None
    
    except Exception as e:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 5.0:
        def local_obj(flat):
            pts = flat.reshape(N, 2)
            if not is_inside(pts, 1e-8):
                return 100.0
            return -min_triangle_area(pts)
        
        if best_points is not None:
            res_local = minimize(
                local_obj, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_local.fun < -best_score:
                best_points = res_local.x.reshape(N, 2)
                best_score = -res_local.fun
    
    # Phase 3: Multiple random restarts with local optimization
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        
        for restart in range(5):
            if time.time() >= deadline - 5.0:
                break
            
            # Generate random points using barycentric coordinates
            uvw = rng.dirichlet([1, 1, 1], size=N)
            p_rand = uvw[:, 1:2] * np.array([[1, 0]]) + uvw[:, 2:3] * np.array([[0.5, H]])
            
            res_local = minimize(
                local_obj, 
                p_rand.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            
            if res_local.fun < -best_score:
                best_points = res_local.x.reshape(N, 2)
                best_score = -res_local.fun
    
    # Final polish if time allows
    if time.time() < deadline - 2.0 and best_points is not None:
        def final_obj(flat):
            pts = flat.reshape(N, 2)
            if not is_inside(pts, 1e-8):
                return 100.0
            return -min_triangle_area(pts)
        
        res_final = minimize(
            final_obj, 
            best_points.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        
        if res_final.fun < -best_score:
            best_points = res_final.x.reshape(N, 2)
            best_score = -res_final.fun
    
    # Fallback: return random valid points if all else fails
    if best_points is None or not is_inside(best_points):
        rng = np.random.default_rng(0)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts.append([x, y])
        best_points = np.array(pts)
    
    return best_points
