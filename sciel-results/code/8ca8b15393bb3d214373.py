import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

N = 50

def construct():
    # Time limit guard
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    # Bounds for differential evolution
    # Centers must be roughly in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def objective(params):
        centers = params.reshape(-1, 2)
        
        # Check if any center is outside the unit circle
        dists_center = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dists_center)
        
        if max_dist > 1.0:
            # Strong penalty for being outside
            # We want to minimize this, so return a large positive number
            # The penalty should scale with how far outside
            return 10.0 + (max_dist - 1.0)
        
        # Calculate minimum pairwise distance
        # pdist is faster than manual loops
        if N > 1:
            d = pdist(centers)
            min_pair_dist = np.min(d)
        else:
            min_pair_dist = np.inf
            
        min_boundary_dist = np.min(1.0 - dists_center)
        
        # The radius of the circles is half the min pairwise distance
        # and also limited by the distance to the boundary
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # We want to maximize r, so minimize -r
        return -r

    # Run Differential Evolution
    # popsize=15 means 15 * 100 = 1500 individuals
    # This might be too slow for many iterations. 
    # Let's start with popsize=10 and see.
    # maxiter=100 might be too many if eval is slow, but pdist is fast.
    
    # We will run DE for a portion of the time, then polish.
    # Let's set a maxiter that is likely to finish in ~80s.
    # Each eval: ~50 points, pdist ~ 1225 ops. Very fast.
    # 1000 evals/sec is conservative. 10000 evals in 1s? 
    # Let's try maxiter=200, popsize=10. 2000 evals.
    
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=10, 
            tol=1e-8, 
            seed=42,
            polish=False # We will do our own polishing
        )
    except Exception as e:
        # Fallback to a simple heuristic if DE fails
        # Hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        else:
            # If still not enough, just return random points (should not happen)
            return np.random.uniform(-0.8, 0.8, (N, 2))

    best_params_de = res_de.x
    best_val_de = res_de.fun
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 20.0:
        # Not enough time for polishing, return DE result
        return best_params_de.reshape(-1, 2)

    # Polish with a local optimizer
    # Nelder-Mead is robust for non-differentiable objectives (like min distance)
    # BFGS requires gradients, which are tricky with min().
    # Let's use Nelder-Mead.
    
    try:
        res_local = minimize(
            objective, 
            best_params_de, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        # Check time again
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
             # Interrupt if too close to limit? 
             # The minimize call is blocking. We can't easily interrupt it.
             # We should have limited maxiter more.
             pass

        if res_local.fun < best_val_de:
            return res_local.x.reshape(-1, 2)
        else:
            return best_params_de.reshape(-1, 2)
            
    except Exception:
        return best_params_de.reshape(-1, 2)
