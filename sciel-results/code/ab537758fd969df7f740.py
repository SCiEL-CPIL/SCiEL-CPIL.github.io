import numpy as np
from scipy.optimize import minimize
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def objective(thetas):
        log_prod = 0.0
        n = len(thetas)
        # Distances within the first half (i < j)
        # These are duplicated in the second half
        sum_within = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(np.exp(1j * thetas[i]) - np.exp(1j * thetas[j]))
                if d < 1e-12:
                    return 1e10
                sum_within += np.log(d)
        
        # Distances between halves
        sum_between = 0.0
        for i in range(n):
            for k in range(n):
                d = abs(np.exp(1j * thetas[i]) + np.exp(1j * thetas[k]))
                if d < 1e-12:
                    return 1e10
                sum_between += np.log(d)
                
        log_prod = 2.0 * sum_within + sum_between
        return -log_prod

    # Initial guess: regular octagon angles for the first 4 points
    # 0, pi/4, pi/2, 3pi/4
    x0_base = np.array([0.0, np.pi/4, np.pi/2, 3.0*np.pi/4])
    
    best_score = np.inf
    best_thetas = x0_base
    
    # Evaluate base
    score = objective(x0_base)
    best_score = score
    best_thetas = x0_base
        
    # Try a few random perturbations to escape local minima (if any)
    # The landscape is likely unimodal near the regular octagon, but restarts are cheap.
    np.random.seed(42)
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the regular octagon
        noise = np.random.normal(0, 0.2, size=4)
        x0 = x0_base + noise
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            score = res.fun
            if score < best_score:
                best_score = score
                best_thetas = res.x
        except Exception:
            pass

    # Construct points from best_thetas
    points = []
    for theta in best_thetas:
        x = math.cos(theta)
        y = math.sin(theta)
        points.append([float(x), float(y)])
        points.append([float(-x), float(-y)])
        
    return points
