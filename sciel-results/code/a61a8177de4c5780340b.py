import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    n = 8
    
    def get_points(r1, r2, theta):
        """
        Generate 8 points as two concentric squares.
        Square 1: radius r1, angle offset 0.
        Square 2: radius r2, angle offset theta.
        Points are (r1*cos(k*pi/2), r1*sin(k*pi/2)) and 
                  (r2*cos(k*pi/2 + theta), r2*sin(k*pi/2 + theta)) for k=0,1,2,3.
        """
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])

    def score(points):
        """
        Compute the score: Delta / n^n where Delta is product of distances
        after scaling to diameter 2.
        """
        # Calculate all pairwise distances
        dists = []
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d)
        
        if len(dists) == 0:
            return 0.0
        
        max_d = max(dists)
        if max_d < 1e-12:
            return 0.0
            
        # Scale to diameter 2
        scale = 2.0 / max_d
        dists_scaled = [d * scale for d in dists]
        
        # Product of distances
        prod = 1.0
        for d in dists_scaled:
            prod *= d
            
        # Score
        return prod / (8.0 ** 8)

    def objective(params):
        """
        Objective function to minimize: -score(params)
        """
        r1, r2, theta = params
        if r1 < 0 or r2 < 0:
            return 1e9
        pts = get_points(r1, r2, theta)
        s = score(pts)
        return -s

    # Strategy:
    # 1. Global Grid Search over r1, r2, theta
    #    r1, r2 in [0.1, 1.5] (relative scales, diameter will be normalized anyway, 
    #    but relative ratio matters. Actually, since we scale to diameter 2, 
    #    absolute scale doesn't matter, only r1/r2 and theta.
    #    Wait, if I fix r1=1, then I only need to search r2 and theta.
    #    Let's fix r1 = 1.0 to reduce dimensionality further.
    #    Then search r2 in [0, 2] and theta in [0, pi/4] (symmetry of squares).
    
    # Fix r1 = 1.0
    # Search r2 in [0, 2.0]
    # Search theta in [0, pi/4]
    
    best_score_val = -1e9
    best_params = (1.0, 1.0, np.pi/4) # Start with regular octagon-like (r2=1, theta=pi/4 is actually a regular octagon? 
    # If r1=1, r2=1, theta=pi/4:
    # P1: (1,0), (0,1), (-1,0), (0,-1)
    # P2: (cos(pi/4), sin(pi/4)), ... which is (0.707, 0.707)...
    # This is exactly the regular octagon. Score should be 1.
    
    # Let's verify regular octagon score quickly in head?
    # Distances in regular octagon side s. 
    # Product of all pairwise distances.
    # The task says regular octagon scores exactly 1.
    
    # Grid Search
    # r2: 0 to 2.0, step 0.05 -> 40 steps
    # theta: 0 to pi/4, step 0.02 -> ~39 steps
    # Total ~1560 evaluations. Very fast.
    
    r2_vals = np.linspace(0.1, 2.0, 40)
    theta_vals = np.linspace(0.0, np.pi/4, 40)
    
    for r2 in r2_vals:
        for theta in theta_vals:
            params = (1.0, r2, theta)
            s = -objective(params)
            if s > best_score_val:
                best_score_val = s
                best_params = params

    # Local Refinement
    # Use the best grid point as initial guess for Nelder-Mead
    # We optimize r1, r2, theta. But since scale is invariant, 
    # we can fix r1=1 and optimize r2, theta? 
    # No, better to optimize all 3 to let it find the optimal relative scaling 
    # if my assumption about fixing r1=1 is too restrictive due to numerical issues, 
    # though mathematically r1=1 is WLOG. 
    # Let's stick to optimizing r2 and theta with r1 fixed to 1 to ensure convergence 
    # in the reduced space, or just optimize all 3 with a constraint? 
    # Simpler: Optimize (r2, theta) with r1=1 fixed.
    
    def obj_2d(x):
        r2, theta = x
        return objective((1.0, r2, theta))
    
    x0 = best_params[1:]
    
    # Run multiple local refinements from nearby points to be safe
    # or just one good one.
    
    try:
        res = minimize(obj_2d, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
        if res.fun < objective(x0): # If it improved
            best_params = (1.0, res.x[0], res.x[1])
            best_score_val = -res.fun
    except Exception:
        pass
    
    # Also try a few random restarts in the 2D space if time permits
    if time.time() < deadline - 5:
        np.random.seed(42)
        for _ in range(50):
            r2_rand = np.random.uniform(0.1, 2.0)
            theta_rand = np.random.uniform(0.0, np.pi/4)
            x_rand = [r2_rand, theta_rand]
            
            try:
                res_r = minimize(obj_2d, x_rand, method='Nelder-Mead', 
                                 options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
                s_r = -res_r.fun
                if s_r > best_score_val:
                    best_score_val = s_r
                    best_params = (1.0, res_r.x[0], res_r.x[1])
            except Exception:
                pass
            
            if time.time() > deadline - 2:
                break

    # Construct final points
    r1, r2, theta = best_params
    pts = get_points(r1, r2, theta)
    
    # Final scale to diameter 2
    max_d = 0
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_d:
                max_d = d
    if max_d > 1e-12:
        pts = pts * (2.0 / max_d)
        
    return [[float(x), float(y)] for x, y in pts]
