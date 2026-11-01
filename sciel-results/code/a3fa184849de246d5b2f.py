import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    def get_score(pts):
        # pts is shape (N, 2)
        try:
            # Check for degenerate cases
            if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                return 0.0
            
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
            
            # Calculate all triangle areas
            min_area = float('inf')
            
            # Vectorized calculation for speed?
            # Let's try a simple loop first, it's small N.
            # To speed up, we can precompute differences?
            # Or just use the loop.
            
            for i in range(N):
                pi = pts[i]
                for j in range(i + 1, N):
                    pj = pts[j]
                    v1 = pj - pi
                    for k in range(j + 1, N):
                        pk = pts[k]
                        v2 = pk - pi
                        # Cross product z-component
                        cross = v1[0] * v2[1] - v1[1] * v2[0]
                        area = 0.5 * abs(cross)
                        if area < min_area:
                            min_area = area
                            if min_area < 1e-12:
                                return 0.0
            return min_area / hull_area
        except Exception:
            return 0.0

    def objective(x):
        pts = x.reshape((N, 2))
        return -get_score(pts)

    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_x = initial_pts.flatten()
    
    best_score = -objective(initial_x)
    best_x = initial_x.copy()
    
    # Optimization settings
    # Nelder-Mead is robust for this size
    options = {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
    
    # Run from regular polygon
    try:
        res = minimize(objective, initial_x, method='Nelder-Mead', options=options)
        if res.fun < -best_score:
            best_score = -res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Try perturbations
    rng = np.random.default_rng(42)
    
    for i in range(10):
        if time.time() > deadline:
            break
        
        # Random perturbation
        noise = rng.normal(0, 0.05, size=N*2)
        x0 = initial_x + noise
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
            if res.fun < -best_score:
                best_score = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    best_pts = best_x.reshape((N, 2))
    return best_pts
