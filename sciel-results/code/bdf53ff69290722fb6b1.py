import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    # Deadline guard: leave 10s for cleanup and return
    deadline = start_time + 140.0

    def objective(x):
        # Check time
        if time.time() > deadline:
            return 1e10  # Large penalty to discourage this region? Or just return current best?
            # Actually, if we return 1e10, it might mess up the convergence.
            # Better to just rely on maxiter.
            
        points = x.reshape(8, 2)
        
        # Calculate pairwise distances
        # diff[i, j, k] = points[i, k] - points[j, k]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Extract upper triangle (i < j) to avoid duplicates and diagonal
        # 8 points -> 28 pairs
        i, j = np.triu_indices(8, k=1)
        d = dists[i, j]
        
        # Check for zero distances (distinct points)
        if np.any(d < 1e-10):
            return 1e10
            
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Scale to diameter 2
        # d_scaled = 2 * d / D
        # log_sum = sum(log(d_scaled)) = sum(log(d)) - sum(log(D/2))
        # = sum(log(d)) - 28 * log(D/2)
        # This is more efficient than creating the array d_scaled
        
        log_d = np.log(d)
        log_D = np.log(D)
        
        # Sum of logs of scaled distances
        # d_scaled = d * (2/D)
        # log(d_scaled) = log(d) + log(2) - log(D)
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        
        sum_log_d = np.sum(log_d)
        constant = 28.0 * (np.log(2.0) - log_D)
        
        total_log = sum_log_d + constant
        
        return -total_log

    # Bounds
    bounds = [(-10.0, 10.0)] * 16
    
    # Initial guess: Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_x = x0
    best_val = objective(x0)
    
    try:
        # Differential Evolution
        # maxiter=300, popsize=15
        # 300 * 15 * 16 = 72,000 evaluations
        # Should be fast.
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=15, 
            tol=1e-8, 
            seed=42, 
            init='sobol'
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Polish with Nelder-Mead if time permits
        if time.time() < deadline - 10.0:
            res2 = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception:
        pass

    best_points = best_x.reshape(8, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
