import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 1e10
        if d_min == 0:
            return 1e10
            
        ratio = d_min / d_max
        return 1.0 / (ratio ** 2)
        
    def get_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0: return 0
        return (d_min / d_max) ** 2

    # Bounds for the search space
    # We can normalize the scale arbitrarily, but keeping it in [-1, 1] helps stability
    bounds = [(-1.0, 1.0)] * (n * 2)
    
    best_score = -1
    best_points = None
    
    # Run Differential Evolution with multiple seeds/restarts
    # DE is stochastic, so running it a few times or with high maxiter is good.
    # We have 600s. DE is fast per function eval, but many evals.
    # Let's try 2-3 runs of DE with moderate iterations, then polish.
    
    start_time = time.time()
    
    # Strategy 1: Differential Evolution
    # We use 'best1bin' mutation and crossover
    # popsize=15 is default, let's try 20 for better coverage
    # maxiter=200 is a reasonable start, we can increase if time permits
    
    for run in range(3):
        if time.time() - start_time > 500:
            break
            
        try:
            # Seed for reproducibility within the run, but different for each run
            seed = 42 + run
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=300, 
                popsize=20, 
                tol=1e-10, 
                seed=seed,
                init='latinhypercube',
                polish=True, # This will run Nelder-Mead on the best result
                update_interval=100
            )
            
            candidate_points = res.x.reshape(n, 2)
            candidate_score = get_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
        except Exception as e:
            pass

    # Strategy 2: Local Polish with Nelder-Mead from best DE result
    if best_points is not None:
        if time.time() - start_time < 550:
            try:
                # Flatten best points
                x0 = best_points.flatten()
                # Use tight tolerances
                res_local = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                local_points = res_local.x.reshape(n, 2)
                local_score = get_score(local_points)
                
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_points.copy()
            except Exception as e:
                pass

    # Fallback if something went wrong (shouldn't happen)
    if best_points is None:
        # Simple grid fallback
        points = np.zeros((n, 2))
        idx = 0
        for j in range(4):
            for i in range(4):
                points[idx, 0] = i
                points[idx, 1] = j
                idx += 1
        points -= np.mean(points, axis=0)
        best_points = points

    return best_points
