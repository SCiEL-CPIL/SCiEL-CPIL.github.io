import numpy as np
import time
import math

N = 21

def _compute_value(centers, W, H):
    """
    Compute the sum of radii for a given configuration of centers in a box W x H.
    Radii are determined by the Voronoi-like constraint:
    r_i = min(boundary_dist_i, 0.5 * min_{j!=i} dist(i, j))
    This guarantees disjoint interiors.
    """
    radii = np.zeros(N)
    for i in range(N):
        # Boundary distance
        r_bound = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        
        # Neighbor distance
        min_dist = float('inf')
        for j in range(N):
            if i == j:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
        
        r_neigh = 0.5 * min_dist
        radii[i] = min(r_bound, r_neigh)
        
    return np.sum(radii)

def _md_simulate(W, H, initial_centers, time_limit=10.0, seed=0):
    """
    Run a Molecular Dynamics simulation to optimize the packing.
    Uses a soft repulsive force to push circles apart and a cooling schedule.
    Returns the best configuration found.
    """
    rng = np.random.RandomState(seed)
    
    # State
    centers = initial_centers.copy()
    velocities = np.zeros_like(centers)
    
    # Parameters
    dt = 0.01
    k_rep = 1000.0  # Repulsion strength
    damping = 0.95  # Velocity damping
    t_start = time.time()
    
    best_value = -1.0
    best_centers = centers.copy()
    
    # Cooling schedule
    # Start with high "temperature" (high noise) and cool down
    # We simulate this by adding random forces that decrease over time
    n_steps = 2000
    for step in range(n_steps):
        if time.time() - t_start > time_limit:
            break
            
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Repulsive forces between circles
        # We want to maximize min distance. 
        # Force is proportional to overlap or inverse distance squared?
        # Since we use Voronoi radii, the objective is non-smooth.
        # A standard trick: Maximize sum of min(dist/2, boundary).
        # Equivalent to: Push centers apart until they hit boundaries or each other.
        
        # Simplified MD: 
        # If circles are "close", push them apart.
        # "Close" is defined by a threshold related to expected radius.
        # Let's use a force that pushes all pairs apart with strength ~ 1/d^2.
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d_sq = dx*dx + dy*dy
                d = math.sqrt(d_sq)
                if d < 1e-6:
                    # Random kick if overlapping exactly
                    dx = rng.randn() * 0.01
                    dy = rng.randn() * 0.01
                    d = math.sqrt(dx*dx + dy*dy)
                
                # Force magnitude
                # We want to maximize the minimum distance.
                # A potential energy U = -1/d works well for maximizing separation.
                # Force = -dU/dr = -1/d^2. Direction: away from each other.
                # So force on i is + (1/d^2) * (dx/d, dy/d)
                
                # To avoid singularities, cap the force
                if d < 0.05:
                    f_mag = 1.0 / (0.05 * 0.05)
                else:
                    f_mag = 1.0 / d_sq
                
                fx = f_mag * dx / d
                fy = f_mag * dy / d
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        # Boundary forces
        # Push centers away from walls to allow them to be as large as possible?
        # Actually, for Voronoi radii, being close to a wall is good IF the neighbor is further.
        # But generally, we want to spread them out.
        # A simple boundary repulsion helps prevent sticking to edges if not optimal.
        # However, optimal packings often have circles touching walls.
        # Let's use a softer boundary force that only activates if center is too close to edge 
        # relative to a "target" radius? 
        # Better: Just let the inter-circle forces do the work, and clamp positions.
        # But if we clamp, velocity is lost. 
        # Let's add a weak centering force to keep them in the box if they drift?
        # No, the box is fixed.
        
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Add thermal noise (annealing)
        # Noise amplitude decreases over time
        temp = 0.5 * (1.0 - step / n_steps)
        velocities += rng.randn(N, 2) * temp * 0.1
        
        # Update positions
        centers += velocities * dt
        
        # Constrain to box [0, W] x [0, H]
        # If a center goes outside, it's invalid for the "inside" constraint 
        # unless we interpret the problem as circles must be inside.
        # The task says "bounding rectangle has perimeter at most 4" and "disjoint interiors".
        # It implies circles are inside the rectangle.
        # So centers must be in [0, W] x [0, H].
        # If a center goes out, we clamp it and zero the velocity component.
        
        for i in range(N):
            if centers[i, 0] < 0:
                centers[i, 0] = 0
                velocities[i, 0] = 0
            elif centers[i, 0] > W:
                centers[i, 0] = W
                velocities[i, 0] = 0
                
            if centers[i, 1] < 0:
                centers[i, 1] = 0
                velocities[i, 1] = 0
            elif centers[i, 1] > H:
                centers[i, 1] = H
                velocities[i, 1] = 0
        
        # Evaluate current state
        val = _compute_value(centers, W, H)
        if val > best_value:
            best_value = val
            best_centers = centers.copy()
            
    return best_centers, best_value

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Search over aspect ratios
    # W + H = 2.0
    # Symmetric, so W in [1.0, 1.5] covers all unique aspect ratios.
    # However, optimal aspect ratio for 21 circles might not be 1.0.
    # Let's sample a range.
    aspect_ratios = np.linspace(0.9, 1.1, 5) # Fine search around 1.0 initially?
    # Actually, let's be broader first, then refine.
    # 21 circles ~ 5x4 or 4x5 grid. 
    # If square, 5x4 grid in 1x1 box -> spacing 0.2. Radii ~0.1. Sum ~2.1.
    # If rectangular, might be better.
    
    # Strategy: 
    # 1. Coarse search over aspect ratios with short MD runs.
    # 2. Fine search around best aspect ratio with longer MD runs.
    
    # Phase 1: Coarse
    coarse_aspects = np.linspace(0.8, 1.2, 9)
    
    for W in coarse_aspects:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Multiple restarts
        for seed in range(3):
            if time.time() > deadline:
                break
            
            # Initial guess: Random
            np.random.seed(seed + int(W*1000))
            x0 = np.random.rand(N, 2)
            x0[:, 0] *= W
            x0[:, 1] *= H
            
            # Run MD with limited time
            t_limit = 5.0
            centers, val = _md_simulate(W, H, x0, time_limit=t_limit, seed=seed)
            
            if val > best_value:
                best_value = val
                centers_best = centers.copy()
                best_W = W
                best_H = H
                
                # Build result
                radii = np.zeros(N)
                for i in range(N):
                    r_bound = min(centers_best[i, 0], W - centers_best[i, 0], centers_best[i, 1], H - centers_best[i, 1])
                    min_dist = float('inf')
                    for j in range(N):
                        if i == j: continue
                        dx = centers_best[i, 0] - centers_best[j, 0]
                        dy = centers_best[i, 1] - centers_best[j, 1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < min_dist: min_dist = d
                    radii[i] = min(r_bound, 0.5 * min_dist)
                    
                result = np.zeros((N, 3))
                result[:, 0] = centers_best[:, 0]
                result[:, 1] = centers_best[:, 1]
                result[:, 2] = radii
                best_result = result

    # Phase 2: Fine
    if best_result is not None:
        # Refine around best_W
        fine_aspects = np.linspace(best_W - 0.1, best_W + 0.1, 5)
        for W in fine_aspects:
            if W <= 0 or W >= 2: continue
            if time.time() > deadline:
                break
            H = 2.0 - W
            
            for seed in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(seed + int(W*10000))
                x0 = np.random.rand(N, 2)
                x0[:, 0] *= W
                x0[:, 1] *= H
                
                t_limit = 10.0
                centers, val = _md_simulate(W, H, x0, time_limit=t_limit, seed=seed)
                
                if val > best_value:
                    best_value = val
                    centers_best = centers.copy()
                    
                    radii = np.zeros(N)
                    for i in range(N):
                        r_bound = min(centers_best[i, 0], W - centers_best[i, 0], centers_best[i, 1], H - centers_best[i, 1])
                        min_dist = float('inf')
                        for j in range(N):
                            if i == j: continue
                            dx = centers_best[i, 0] - centers_best[j, 0]
                            dy = centers_best[i, 1] - centers_best[j, 1]
                            d = math.sqrt(dx*dx + dy*dy)
                            if d < min_dist: min_dist = d
                        radii[i] = min(r_bound, 0.5 * min_dist)
                        
                    result = np.zeros((N, 3))
                    result[:, 0] = centers_best[:, 0]
                    result[:, 1] = centers_best[:, 1]
                    result[:, 2] = radii
                    best_result = result

    if best_result is None:
        # Fallback: Grid
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) / cols
            y[i] = (r_idx + 0.5) / rows
            r[i] = 0.09
        best_result = np.column_stack((x, y, r))
        
    return best_result
