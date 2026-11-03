import numpy as np
from scipy.optimize import basinhopping
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # We use Basinhopping to find a near-optimal packing.
    # The objective is to maximize the minimum radius.
    # r = min(min_pairwise_dist / 2, min_boundary_dist)
    # We minimize -r.
    
    def objective(x):
        # x is a flat array of shape (2*N,)
        points = x.reshape((N, 2))
        
        # Calculate pairwise distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        min_pairwise = np.min(dists)
        
        # Distance to boundary
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii) # Wait, we need min(1 - |c_i|)
        # Actually, the constraint is 1 - |c_i|. The limiting factor is the max |c_i|.
        # So min_boundary_dist = 1.0 - np.max(np.linalg.norm(points, axis=1))
        
        # The common radius is limited by half the min pairwise distance
        # and the distance to the boundary.
        r = min(min_pairwise / 2.0, 1.0 - np.max(radii))
        
        # We want to maximize r, so we minimize -r
        # Add a small penalty if points are too close to ensure numerical stability?
        # No, -r is sufficient.
        
        return -r

    # Initial guess: Hexagonal lattice
    # This provides a good starting point to avoid getting stuck in bad local minima.
    def initial_guess():
        # Generate a hex lattice and clip to unit circle
        spacing = 0.25
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2.0
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback to random if lattice is too sparse (unlikely with this spacing)
            return np.random.uniform(-1, 1, (N, 2))
        
        # Select N points closest to center? Or just first N?
        # Let's select N points with max min-distance? 
        # For simplicity, just take the first N.
        # Better: Sort by distance to center and take N?
        # Actually, for a good start, we want them spread out.
        # Let's just use the first N from the lattice.
        return points[:N].flatten()

    x0 = initial_guess()
    
    # Define the local optimizer
    # Nelder-Mead is good for non-smooth functions
    local_min = {'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'maxfev': 10000}}
    
    # Basinhopping parameters
    # stepsize: size of the random perturbation
    # mininterval, maxinterval: interval between global moves
    # niter: number of global iterations
    
    # We have 120 seconds. Let's monitor time.
    start_time = time.time()
    
    try:
        # Run basinhopping
        # We might not be able to run for a very long time.
        # Let's run a limited number of iterations.
        res = basinhopping(objective, x0, niter=50, stepsize=0.5, 
                           minimizer_kwargs=local_min)
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed < 100:
            # If we have time left, run another round with a different start or more iterations
            # Or just use the result.
            pass
        else:
            # If timeout, we might need to break early, but basinhopping doesn't have a callback to break easily 
            # unless we wrap the objective.
            pass
            
        x_best = res.x
        r_best = -res.fun
        
    except Exception as e:
        # Fallback to the initial guess if optimization fails
        x_best = x0
        r_best = -objective(x0)
        
    # Verify validity
    points = x_best.reshape((N, 2))
    # Ensure points are within bounds (should be, but just in case)
    radii = np.linalg.norm(points, axis=1)
    mask = radii <= 1.0
    if not np.all(mask):
        # Project points back inside
        points[~mask] = points[~mask] / radii[~mask][:, np.newaxis] * 0.999
        
    return points

# EVOLVE-BLOCK-END
