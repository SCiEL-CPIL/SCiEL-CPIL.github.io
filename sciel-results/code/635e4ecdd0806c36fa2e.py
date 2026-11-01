import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.zeros(2 * N)
    x0[0::2] = np.cos(angles)
    x0[1::2] = np.sin(angles)
    
    def get_points(x):
        return x.reshape(N, 2)
    
    def objective(x):
        if time.time() > deadline:
            return 0.0 # Signal timeout, though minimize doesn't handle this well directly, 
                       # we will check after or use a wrapper if needed. 
                       # For Powell, it's better to just let it run and check at the end.
        
        pts = get_points(x)
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
            if hull_area < 1e-10:
                return 1e10 # Penalize degenerate hull
        except Exception:
            return 1e10 # Penalize invalid hull
            
        # Compute all triangle areas
        # Efficient way: 
        # Area = 0.5 * | (p1-p0) x (p2-p0) |
        # We can vectorize this.
        # Get all combinations of 3 indices
        # i, j, k
        # To avoid itertools overhead in tight loop, we can precompute indices or use meshgrid
        # But 364 triangles is small. It's fine.
        
        # Vectorized triangle area calculation
        # i, j, k ranges
        # Let's use a simple loop for clarity and speed is likely sufficient for 364 items
        # Or better, vectorized:
        # p_i, p_j, p_k
        # Cross product in 2D: (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
        
        # Precompute all pairs? No, just do it directly.
        # Create indices for all triples
        # This creates a 364x3 array
        # idx = np.array(list(itertools.combinations(range(N), 3)))
        # This is slow to create every time. Create it once outside.
        
        pass # Placeholder
        
    # Precompute indices for all triples
    import itertools
    triples = np.array(list(itertools.combinations(range(N), 3)))
    
    def objective_fast(x):
        pts = x.reshape(N, 2)
        
        # Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e10
        except Exception:
            return 1e10
            
        # Triangle Areas
        # p_i = pts[triples[:, 0]]
        # p_j = pts[triples[:, 1]]
        # p_k = pts[triples[:, 2]]
        
        pi = pts[triples[:, 0]]
        pj = pts[triples[:, 1]]
        pk = pts[triples[:, 2]]
        
        # Area = 0.5 * | (pj - pi) x (pk - pi) |
        v1 = pj - pi
        v2 = pk - pi
        # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return -(min_area / hull_area)
        
    # Optimization
    # Powell is good for non-smooth. 
    # Nelder-Mead is also good.
    # Let's try Powell.
    
    best_x = x0
    best_val = objective_fast(x0)
    
    try:
        # Use Powell method
        # maxiter limit to ensure we don't run too long
        # 28 variables. 1000 iterations is usually enough for convergence or plateau.
        res = minimize(objective_fast, x0, method='Powell', 
                       options={'maxiter': 2000, 'maxfev': 50000})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception:
        pass
    
    # Check if we have time for a second polish with Nelder-Mead or another method?
    # Powell is usually robust.
    
    if time.time() < deadline - 5:
        # Try a local refinement with Nelder-Mead starting from the Powell result
        try:
            res2 = minimize(objective_fast, best_x, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass
            
    return best_x.reshape(N, 2)
