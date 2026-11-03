import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    def hexagonal_initial_guess():
        # Create a hexagonal grid
        points = []
        # Approximate radius for N=50 is ~0.126
        # Spacing should be slightly larger than 2*r to allow optimization
        # Let's start with a dense grid and let optimization expand it
        spacing = 0.15 
        # Generate points in a large enough area
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing * 1.732
                y = j * spacing + (i % 2) * spacing * 0.5 * 1.732
                if np.hypot(x, y) < 0.9:
                    points.append([x, y])
        
        points = np.array(points)
        # If we have too many, subsample or just take the first N
        # If we have too few, this guess is bad. 
        # For N=50, a grid of spacing 0.15 in radius 0.9 should have plenty.
        # Area of circle r=0.9 is ~2.54. Area of hex cell ~ 0.15^2 * 1.732/2 ~ 0.02.
        # 2.54 / 0.02 ~ 127 points. So we have enough.
        
        # Select N points closest to center to start? Or random?
        # Closest to center is a good start for a "packed" look.
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)[:N]
        return points[indices]

    def energy(x):
        """
        Objective function to minimize.
        We want to maximize the minimum distance between circles and to the boundary.
        Minimizing the sum of inverse squared distances achieves this.
        """
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        # Efficient calculation of all pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0) # Diagonal is 0, we don't care about self-distance
        
        # We only care about unique pairs, but summing all is fine (just doubles the weight)
        # Inverse square potential
        # Add epsilon to prevent infinity
        eps = 1e-6
        pair_energy = np.sum(1.0 / (dists + eps)**2)
        
        # Boundary energy
        # Distance from center
        center_dists = np.sqrt(np.sum(x**2, axis=1))
        # Distance to boundary
        boundary_dists = 1.0 - center_dists
        # If a point is outside, boundary_dist is negative.
        # We want to penalize being close to boundary (small boundary_dist) or outside.
        # Using 1/(boundary_dist + eps)^2 will penalize small positive values.
        # If boundary_dist is negative, (neg + eps)^2 is small positive, so 1/small is large.
        # This effectively pushes points inside.
        
        # However, if a point is far outside, boundary_dist is very negative.
        # ( -10 + eps )^2 is 100. 1/100 is small.
        # So it doesn't penalize being far outside as much as being close to the boundary.
        # This is okay, as the pairwise repulsion will also push them around.
        # But to be safe, we can clamp boundary_dists to a small positive value for the denominator?
        # No, if it's outside, we want a huge penalty.
        # Let's use a soft plus:
        # penalty = 1 / (max(boundary_dist, eps) + eps)^2 ?
        # If boundary_dist < 0, max is eps. penalty = 1/(2eps)^2. Huge.
        # If boundary_dist > 0, penalty = 1/(bd+eps)^2.
        
        safe_boundary_dists = np.maximum(boundary_dists, eps)
        boundary_energy = np.sum(1.0 / (safe_boundary_dists + eps)**2)
        
        return pair_energy + boundary_energy

    best_centers = None
    best_radius = 0.0
    
    # Initial guess
    x0 = hexagonal_initial_guess()
    
    # Run optimization with multiple restarts/jitters
    # L-BFGS-B is good for this
    options = {
        'maxiter': 200,
        'ftol': 1e-10,
        'gtol': 1e-8
    }
    
    # We will try a few variations
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Jitter the initial guess
        # Small random perturbation
        jitter = np.random.normal(0, 0.01, x0.shape)
        x_init = x0 + jitter
        
        # Ensure initial points are not exactly on top of each other
        # L-BFGS-B might struggle if starting from exact duplicates, but jitter helps
        
        try:
            res = minimize(energy, x_init, method='L-BFGS-B', options=options)
            x_opt = res.x.reshape(-1, 2)
            
            # Calculate the actual radius for this configuration
            # r = min( min_{i<j} d_ij / 2, min_i (1 - ||x_i||) )
            
            # Pairwise distances
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Boundary distances
            center_dists = np.sqrt(np.sum(x_opt**2, axis=1))
            min_boundary_dist = np.min(1.0 - center_dists)
            
            current_radius = min(min_pair_dist / 2.0, min_boundary_dist)
            
            # Check validity: all points must be inside
            if np.all(center_dists <= 1.0) and current_radius > best_radius:
                best_radius = current_radius
                best_centers = x_opt
                
        except Exception:
            continue

    # If no valid center found (should not happen), return the initial guess
    if best_centers is None:
        best_centers = x0
        # Calculate radius for x0 just in case
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        center_dists = np.sqrt(np.sum(best_centers**2, axis=1))
        min_boundary_dist = np.min(1.0 - center_dists)
        best_radius = min(min_pair_dist / 2.0, min_boundary_dist)

    return best_centers
# EVOLVE-BLOCK-END
