import scipy
import numpy as np
import time
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area divided by the convex hull area.
    Uses local optimization starting from a regular 13-gon.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial configuration: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = initial_points
    best_score = -1.0
    
    def calc_score(points):
        # Calculate convex hull area
        # Using scipy.spatial.ConvexHull
        try:
            hull = scipy.spatial.ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0
        
        if hull_area < 1e-9:
            return 0.0
            
        # Calculate minimum triangle area
        min_area = float('inf')
        # Iterate over all triples
        # N=13, so 13*12*11/6 = 286 triangles. Fast.
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * | (p2-p1) x (p3-p1) |
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
                        
        return min_area / hull_area

    # Evaluate initial
    score_init = calc_score(initial_points)
    best_score = score_init
    
    if time.time() > deadline:
        return best_points

    # Objective function for minimization (we want to maximize score, so minimize -score)
    def objective(x):
        points = x.reshape((N, 2))
        s = calc_score(points)
        return -s

    # Optimization
    # Use Nelder-Mead as it doesn't require gradients
    # We can try a few restarts with small perturbations
    
    n_restarts = 5
    max_time_per_restart = (deadline - time.time()) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial points slightly
        if r == 0:
            x0 = initial_points.flatten()
        else:
            noise = np.random.normal(0, 0.1, (N, 2))
            x0 = (initial_points + noise).flatten()
            
        # Set a time limit for this specific optimization run
        # SciPy doesn't have a built-in time limit for minimize, 
        # but we can check inside the callback or just rely on the outer loop.
        # To be safe, we can use a callback that raises an exception if time is up.
        
        def callback(xk):
            if time.time() > deadline - 2.0:
                raise TimeoutError()

        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8},
                           callback=callback)
            
            if res.success or res.fun < -best_score:
                current_points = res.x.reshape((N, 2))
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points
                    
        except TimeoutError:
            pass
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
