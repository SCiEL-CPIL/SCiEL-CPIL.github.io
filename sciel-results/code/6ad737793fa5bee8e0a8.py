"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a time-evolutionary optimization with adaptive control (learning rate).
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    def calc_radius(points):
        """Calculates the maximum feasible radius for the given points."""
        # Pairwise distances
        # For N=300, 300x300 is 90,000 ops, very fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        norm_points = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - norm_points)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        return r

    def get_hex_grid_points(n_points):
        """Generates a hexagonal grid of points inside the unit circle."""
        # Estimate spacing to get roughly n_points
        # Area of circle = pi. Area per point = pi/n.
        # Hex grid area per point = s^2 * sqrt(3)/2.
        # s^2 * sqrt(3)/2 = pi/n => s = sqrt(2*pi/(n*sqrt(3)))
        s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        
        # Generate grid
        # Range needs to cover radius 1
        max_r = 1.0
        # Number of steps in x
        nx = int(2 * max_r / s) + 1
        ny = int(2 * max_r / (s * np.sqrt(3)/2)) + 1
        
        xs = np.linspace(-max_r, max_r, nx)
        ys = np.linspace(-max_r, max_r, ny)
        
        points = []
        for i, x in enumerate(xs):
            offset = 0.0 if i % 2 == 0 else s * np.sqrt(3) / 2
            for y in ys:
                y_off = y + offset
                if x**2 + y_off**2 <= 1.0:
                    points.append((x, y_off))
        
        points = np.array(points)
        
        # If we have more than n_points, select the ones closest to the center?
        # Or just return the first n_points?
        # For packing, uniformity is key. The hex grid is uniform.
        # If we have more, we should probably subsample or just take the first N.
        # Taking the first N from a sorted grid might be biased.
        # Let's sort by distance from center and take the first N?
        # Actually, for a packing, we want to fill the space.
        # If the grid is too dense, we have too many points.
        # If the grid is too sparse, we have too few.
        # My s calculation should give approx N points.
        
        if len(points) >= n_points:
            # Select n_points randomly? Or closest to center?
            # Random selection from a good grid is a decent start.
            # Or, to ensure coverage, we can just take the first N if the grid is generated in a spiral?
            # The current generation is row-major.
            # Let's just take the first N, but shuffle the points first to avoid bias?
            # No, the grid is symmetric.
            # Let's sort by distance from center and take the innermost N?
            # This might leave empty space at the boundary.
            # Better: Take a random subset of the grid points?
            # Or, adjust s to get exactly N? Hard.
            # Let's just take the first N. The grid is dense enough.
            indices = np.random.choice(len(points), n_points, replace=False)
            return points[indices]
        else:
            # Pad with random points if too few?
            # Should not happen with the formula.
            while len(points) < n_points:
                # Add random point inside circle
                r = np.sqrt(np.random.random())
                theta = np.random.random() * 2 * np.pi
                points.append((r * np.cos(theta), r * np.sin(theta)))
            return np.array(points)

    def optimize(points, target_time):
        """
        Optimizes the configuration using adaptive repulsion.
        """
        points = points.copy()
        current_r = calc_radius(points)
        best_r = current_r
        best_points = points.copy()
        
        # Adaptive learning rate
        lr = 0.05
        decay = 0.99
        increase_factor = 1.05
        
        while time.time() - start_time < target_time:
            # Calculate forces
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            # Repulsion force: 1/d^2
            # We want to push points apart.
            # Force on i from j: (c_i - c_j) / d_ij^2
            # This is equivalent to gradient of 1/d.
            # To make it stronger when overlapping, we can use 1/d^3 or similar.
            # Let's stick to 1/d^2 for stability.
            
            # Normalize diff to unit vector, then scale by 1/d^2?
            # unit_diff = diff / dists_safe
            # force = unit_diff / dists_safe^2
            # Or simply diff / dists_safe^3
            
            inv_d3 = 1.0 / (dists_safe ** 3)
            # Set diagonal to 0
            np.fill_diagonal(inv_d3, 0.0)
            
            # Force is sum of (c_i - c_j) / d^3
            forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
            
            # Boundary force
            norms = np.linalg.norm(points, axis=1)
            # If point is outside 1 - r_target?
            # We don't have a fixed r_target, we are maximizing r.
            # We want to keep points inside the circle.
            # Force towards center if |c_i| > 0.8 (soft boundary)
            # Or hard constraint: if |c_i| >= 1, push in.
            # Let's use a soft potential: exp(|c_i|) or just linear if > 0.9
            boundary_forces = np.zeros_like(points)
            # Push in if norm > 0.9
            mask = norms > 0.9
            if np.any(mask):
                # Strength proportional to violation
                strength = (norms[mask] - 0.9) ** 2
                # Direction: towards center
                direction = -points[mask] / norms[mask, np.newaxis]
                boundary_forces[mask] = strength[:, np.newaxis] * direction
            
            total_forces = forces + boundary_forces
            
            # Update
            points += lr * total_forces
            
            # Calculate new radius
            new_r = calc_radius(points)
            
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
                # Increase lr slightly if we are making progress
                lr = min(lr * increase_factor, 0.2)
            else:
                # No improvement
                # Decay lr
                lr *= decay
                # If lr is very small, we might be stuck.
                # Add small noise to escape local minima
                if lr < 0.001:
                    points += np.random.normal(0, 0.01, size=points.shape)
                    # Re-evaluate
                    new_r = calc_radius(points)
                    if new_r > best_r:
                        best_r = new_r
                        best_points = points.copy()
                    lr = 0.05 # Reset lr
            
        return best_points, best_r

    # Main loop with restarts
    best_points_overall = None
    best_r_overall = 0.0
    
    # Try different initializations
    inits = [
        get_hex_grid_points(N),
        get_hex_grid_points(N) + np.random.normal(0, 0.05, size=(N, 2)),
        np.array([[np.sqrt(np.random.random()) * np.cos(2*np.pi*np.random.random()), 
                   np.sqrt(np.random.random()) * np.sin(2*np.pi*np.random.random())] for _ in range(N)])
    ]
    
    # Allocate time for each restart
    # If time allows, do 3 restarts.
    # Time per restart: (time_limit - elapsed) / remaining_restarts
    
    for i, init_points in enumerate(inits):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        # Allocate time for this restart
        # Give more time to the first few?
        time_for_this = remaining_time / (len(inits) - i)
        
        # Ensure points are inside
        norms = np.linalg.norm(init_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            init_points[mask] = init_points[mask] / norms[mask, np.newaxis] * 0.99
        
        optimized_points, r = optimize(init_points, time.time() + time_for_this)
        
        if r > best_r_overall:
            best_r_overall = r
            best_points_overall = optimized_points
    
    return best_points_overall

# EVOLVE-BLOCK-END
