"""
Packing 200 equal circles in a unit square using Langevin Dynamics.
Strategy: Treat circles as particles with repulsive forces.
Use a neighbor list (or full N^2 since N=200 is small) to compute forces.
Add stochastic noise (Langevin) to escape local minima.
Cool the system (annealing) to settle into a dense packing.
"""
import numpy as np
import time

N = 200
SIDE = 1.0
HALF_SIDE = 0.5

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0:
        return 0.0
    centers = np.asarray(centers)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -HALF_SIDE - 1e-9) or np.any(centers > HALF_SIDE + 1e-9):
        return 0.0
        
    # Distance to walls
    wall_dist = np.min(np.minimum(np.minimum(HALF_SIDE - centers[:, 0], HALF_SIDE - centers[:, 1]),
                                  np.minimum(centers[:, 0] + HALF_SIDE, centers[:, 1] + HALF_SIDE)))
    
    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    # Estimate d
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 1.0
    
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback to square grid if hex fails to generate enough
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _generate_random_grid(N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, (N, 2))

def _langevin_step(centers, velocities, dt, temp, force_cutoff=0.15):
    """
    Perform one Langevin dynamics step.
    Force: Repulsive 1/r^2 between pairs within cutoff, and strong wall repulsion.
    """
    forces = np.zeros_like(centers)
    
    # Pairwise forces
    # N=200 is small enough for O(N^2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    # Only consider pairs within cutoff to save some time and avoid long-range noise
    mask = dist_sq < force_cutoff**2
    # Set diagonal to 0 in mask
    np.fill_diagonal(mask, False)
    
    # Calculate forces: F = -grad V. V ~ 1/r. F ~ 1/r^2 * r_hat = diff / r^3
    # We use a steeper potential for better packing: V ~ 1/r^10 -> F ~ 1/r^11
    # Let's stick to 1/r^2 for stability, or maybe 1/r^4.
    # Using 1/r^2: F_ij = k * (r_i - r_j) / |r_i - r_j|^3
    # dist_sq is |r_i - r_j|^2. dist = sqrt(dist_sq).
    # F = diff / dist_sq * (1/dist) = diff / dist_sq^1.5
    
    dist = np.sqrt(dist_sq)
    # Avoid division by zero for non-masked elements (though we multiply by mask)
    dist_safe = np.where(dist > 1e-8, dist, 1.0)
    
    # Force magnitude factor: 1/r^2 * direction
    # F = diff / (dist^3)
    force_mag = 1.0 / (dist_safe**3)
    # Apply mask
    force_mag = np.where(mask, force_mag, 0.0)
    
    # Sum forces from all other particles
    # F_i = sum_j F_ij
    forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
    
    # Wall forces: Soft repulsion
    # d = distance to wall
    dx = centers[:, 0]
    dy = centers[:, 1]
    
    # Left wall: x < -0.5
    d_left = centers[:, 0] + HALF_SIDE
    f_left = np.where(d_left < 0.1, (0.1 - d_left)**2 * 100, 0)
    # Right wall: x > 0.5
    d_right = HALF_SIDE - centers[:, 0]
    f_right = np.where(d_right < 0.1, (0.1 - d_right)**2 * 100, 0)
    
    # Bottom wall: y < -0.5
    d_bottom = centers[:, 1] + HALF_SIDE
    f_bottom = np.where(d_bottom < 0.1, (0.1 - d_bottom)**2 * 100, 0)
    # Top wall: y > 0.5
    d_top = HALF_SIDE - centers[:, 1]
    f_top = np.where(d_top < 0.1, (0.1 - d_top)**2 * 100, 0)
    
    forces[:, 0] += -f_left + f_right
    forces[:, 1] += -f_bottom + f_top
    
    # Langevin update
    # v_new = v * exp(-gamma*dt) + sqrt(2*gamma*T*dt) * N(0,1) + F*dt * exp(-gamma*dt)
    # Simplified: v += F*dt - gamma*v*dt + noise
    gamma = 1.0
    noise = np.random.normal(0, np.sqrt(2 * gamma * temp * dt), centers.shape)
    velocities = velocities + (forces - gamma * velocities) * dt + noise
    centers = centers + velocities * dt
    
    return centers, velocities

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # Initializations
    inits = []
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
    inits.append(_generate_square_grid(N))
    inits.append(_generate_random_grid(N, seed=42))
    inits.append(_generate_random_grid(N, seed=123))
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        centers = init_centers.copy()
        velocities = np.zeros_like(centers)
        
        # Annealing schedule
        # Start with high temp to explore, cool down to settle
        initial_temp = 0.005
        final_temp = 0.0001
        num_steps = 5000
        
        # Adaptive dt
        dt = 0.01
        
        for step in range(num_steps):
            if time.time() - start_time > time_limit:
                break
                
            # Linear cooling
            progress = step / num_steps
            temp = initial_temp * (1.0 - progress) + final_temp * progress
            
            # Reduce dt as we cool to settle
            current_dt = dt * (1.0 - 0.5 * progress)
            
            centers, velocities = _langevin_step(centers, velocities, current_dt, temp)
            
            # Optional: Compute score every few steps to track best
            if step % 100 == 0:
                score = _compute_score(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
        
        # Final score for this run
        final_score = _compute_score(centers)
        if final_score > best_score:
            best_score = final_score
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = inits[0]
        
    return best_centers
