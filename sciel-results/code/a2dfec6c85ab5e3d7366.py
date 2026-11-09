import numpy as np
import time

N = 200

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    # 1. Initialization: Hexagonal Grid
    # Estimate spacing for 200 points in area 1
    # Density ~ 200 / 1.0
    # Hex packing density ~ 0.9069
    # Area per circle ~ pi * r^2
    # 200 * pi * r^2 <= 0.9069 * 1
    # r^2 <= 0.9069 / (200 * pi) ~ 0.00144
    # r ~ 0.038
    # Spacing s = 2r ~ 0.076
    
    s = 0.08  # Initial spacing, will be optimized
    
    # Generate hex grid points
    points = []
    y = -0.5
    row_idx = 0
    while y <= 0.5:
        x_offset = (s / 2) if row_idx % 2 == 1 else 0.0
        x = -0.5 + x_offset
        while x <= 0.5:
            points.append((x, y))
            x += s
        y += s * np.sqrt(3) / 2
        row_idx += 1
        
    points = np.array(points)
    
    # Filter points strictly inside (with small margin to avoid boundary issues initially)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, we need to select N.
    # If we have fewer, we need to add random points or adjust grid.
    # For hex grid with s=0.08, we might have more than 200.
    if len(points) > N:
        # Select points closest to the center to form a compact cluster
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        centers = points[indices].copy()
    elif len(points) < N:
        # Pad with random points inside square
        random_points = np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))
        centers = np.vstack([points, random_points])
    else:
        centers = points.copy()

    # Ensure centers are strictly inside bounds (avoid boundary invalidity)
    eps = 1e-6
    centers = np.clip(centers, -0.5 + eps, 0.5 - eps)

    # Function to calculate radius
    def get_radius(c):
        # Boundary distance
        # min(0.5 - |x|, 0.5 - |y|)
        bound_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        min_bound = np.min(bound_dists)
        
        # Pairwise distances
        # Use broadcasting for pairwise differences
        # c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # This creates N x N x 2 array. For N=200, this is 200*200*2*8 bytes = 640KB. Fine.
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return np.min([min_bound, min_pair])

    current_radius = get_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()

    # Simulated Annealing
    T = 0.05  # Initial temperature (relative to radius scale)
    alpha = 0.9995  # Cooling rate
    step_size = 0.05  # Initial move size
    
    # We will run until time limit
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Propose move
        delta = np.random.normal(0, step_size, size=2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Clip to square (strictly inside)
        new_centers[idx] = np.clip(new_centers[idx], -0.5 + eps, 0.5 - eps)
        
        # Calculate new radius
        new_radius = get_radius(new_centers)
        
        # Acceptance probability
        if new_radius > current_radius:
            accept = True
        else:
            # Metropolis criterion
            # exp((new - current) / T)
            # If T is small, this is very small for negative diff
            delta_r = new_radius - current_radius
            if delta_r < 0:
                # Avoid overflow/underflow
                if T > 0:
                    prob = np.exp(delta_r / T)
                else:
                    prob = 0.0
            else:
                prob = 1.0
            
            accept = np.random.random() < prob
            
        if accept:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Revert
            pass # centers unchanged
            
        # Cooling
        T *= alpha
        if T < 1e-5:
            T = 1e-5
            # If temperature is very low, we might be stuck in local min.
            # We can restart with a slightly perturbed best solution or increase step size occasionally.
            # For simplicity, let's just keep going. The random moves might still help if step_size is large enough.
            # Or we can implement a restart strategy.
            
        # Adjust step size: decrease as T decreases
        step_size = 0.01 + 0.04 * (T / 0.05)
        if step_size < 0.001:
            step_size = 0.001

    # Final polish: Local optimization (Gradient Ascent on smooth surrogate)
    # We can do a few iterations of pushing points apart.
    # But time might be tight. The SA should have done the heavy lifting.
    
    return best_centers
