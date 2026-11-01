import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Initialize with regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Circumradius for unit area regular N-gon
    # Area = N/2 * R^2 * sin(2pi/N) = 1 => R = sqrt(2 / (N * sin(2pi/N)))
    R = np.sqrt(2.0 / (N * np.sin(2.0 * np.pi / N)))
    x0 = np.column_stack((R * np.cos(angles), R * np.sin(angles)))
    x0_flat = x0.flatten()
    
    # 2. Define Objective Function
    def objective(flat_x):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop or ignore
            
        pts = flat_x.reshape(N, 2)
        
        # Calculate Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            return 1e10 # Degenerate hull
            
        if hull_area < 1e-6:
            return 1e10
            
        # Calculate all triangle areas
        # We need min area over all triples
        # N=14, C(14,3) = 364 triangles. This is fast.
        min_area = np.inf
        # Use itertools for clarity, or vectorized
        # Vectorized approach for speed:
        # Get all pairs? No, triples.
        # Since N is small, loops are fine, but let's try to be efficient.
        
        # Pre-compute cross products?
        # Area = 0.5 * | (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) |
        
        # Simple loop is likely fast enough for 364 iterations in numpy if vectorized inner?
        # Actually, 364 is very small.
        
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Vectorized area calculation for this triple is overkill, just do it
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                    if area < min_area:
                        min_area = area
                        
        # Objective is to maximize min_area / hull_area
        # So minimize - (min_area / hull_area)
        return -(min_area / hull_area)

    # 3. Optimization
    # Powell is good for non-smooth functions and moderate dimensions
    # Nelder-Mead is also an option
    # Let's try Powell first
    
    best_x = x0_flat
    best_score = -objective(x0_flat)
    
    # Run optimization
    try:
        res = minimize(
            objective, 
            x0_flat, 
            method='Powell',
            options={
                'maxiter': 2000, 
                'maxfev': 20000,
                'xtol': 1e-6,
                'ftol': 1e-6
            }
        )
        
        if res.success or (res.fun < objective(x0_flat)):
            # Check time before accepting
            if time.time() < deadline:
                best_x = res.x
                best_score = -res.fun
    except Exception:
        pass
    
    # If time permits, try a second polish with Nelder-Mead or L-BFGS-B (if smooth approx used)
    # But Powell should be sufficient.
    
    # Ensure we return the best found
    if time.time() > deadline - 2.0:
        # Return current best
        pass
    else:
        # Try a quick local polish with Nelder-Mead starting from best_x
        try:
            res2 = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-6
                }
            )
            if res2.fun < objective(best_x):
                best_x = res2.x
        except Exception:
            pass

    return best_x.reshape(N, 2)
