"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    """
    Pack N equal circles in unit disk using force-directed relaxation.
    """
    # 1. Initialization: Hexagonal grid
    # We want to initialize with a density that is slightly higher than the target
    # to ensure we start with overlaps which the repulsion will resolve.
    
    # Estimate optimal radius
    r_opt_est = 0.0531542991469
    
    # Create a hexagonal grid
    # Spacing s such that we have enough points
    # Area of unit disk = pi. Area per circle ~ pi / N.
    # Hex lattice point density ~ 2 / (sqrt(3) * s^2).
    # We want ~ N points in area pi.
    # N ~ pi / (sqrt(3)/2 * s^2) => s ~ sqrt(2*pi / (sqrt(3)*N))
    s_init = math.sqrt(2 * math.pi / (math.sqrt(3) * N))
    
    # Generate points in a hexagonal grid covering [-1, 1] x [-1, 1]
    # To be safe, generate a grid larger than the unit disk
    max_coord = 1.2
    n_x = int(2 * max_coord / s_init) + 1
    n_y = int(2 * max_coord / (s_init * math.sqrt(3) / 2)) + 1
    
    points = []
    for j in range(n_y):
        for i in range(n_x):
            x = (i - n_x // 2) * s_init
            y = (j - n_y // 2) * s_init * math.sqrt(3) / 2
            if j % 2 == 1:
                x += s_init / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter points inside the unit disk (with a margin)
    dists = np.hypot(points[:, 0], points[:, 1])
    mask = dists <= 1.0 - s_init/2
    points = points[mask]
    
    # If we have too many, subsample or take the first N
    # If we have too few, we need a denser grid
    if len(points) < N:
        # Increase density
        s_init *= 0.9
        return construct() # Recursive retry with denser grid (careful with depth)
        
    # Randomly select N points to avoid symmetry bias? 
    # Or select the ones closest to the center?
    # Random selection from the valid grid points is a good start for force-directed.
    # But we must ensure they are within the container.
    idx = np.random.choice(len(points), N, replace=False)
    pos = points[idx].copy()
    
    # 2. Force-Directed Relaxation
    # We will try to maximize the minimum radius.
    # Strategy: Start with a slightly smaller radius, relax, then increase.
    
    best_pos = pos.copy()
    best_r = 0.0
    
    # Binary search or incremental increase on radius
    r_low = 0.04
    r_high = 0.054
    
    # We can just run relaxation for a target r and see what min radius we get.
    # Let's try a few target radii.
    
    target_rs = [0.050, 0.052, 0.053, 0.0531, 0.05315, 0.05316]
    
    for r_target in target_rs:
        # Initialize positions near the previous best if possible, or re-init
        # Re-initializing with a grid is safer for different r
        pos = _init_hexagonal(N, r_target)
        
        # Relax
        pos = _relax_forces(pos, r_target, iterations=3000, seed=42)
        
        # Compute actual min radius
        r_actual = _compute_min_radius(pos)
        
        if r_actual > best_r:
            best_r = r_actual
            best_pos = pos.copy()
            
    # Final polish with the best found radius
    # Try to improve best_pos by relaxing with r = best_r * 1.001
    # This is optional and might not help if we are already at the limit.
    
    return best_pos

def _init_hexagonal(N, r_target):
    """Initialize positions in a hexagonal grid suitable for r_target."""
    # Spacing based on r_target
    s = 2.0 * r_target * 1.05 # Slightly larger than diameter to start with gaps
    
    # Generate grid
    max_coord = 1.2
    n_x = int(2 * max_coord / s) + 1
    n_y = int(2 * max_coord / (s * math.sqrt(3) / 2)) + 1
    
    points = []
    for j in range(n_y):
        for i in range(n_x):
            x = (i - n_x // 2) * s
            y = (j - n_y // 2) * s * math.sqrt(3) / 2
            if j % 2 == 1:
                x += s / 2
            points.append([x, y])
            
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    mask = dists <= 1.0 - r_target
    points = points[mask]
    
    if len(points) < N:
        # If not enough points, use a denser grid
        s *= 0.95
        return _init_hexagonal(N, r_target)
        
    # Select N points. 
    # To improve packing, we can select points that are "random" but ensure coverage.
    # A simple random selection from the grid works well.
    idx = np.random.choice(len(points), N, replace=False)
    return points[idx].copy()

def _relax_forces(pos, r_target, iterations=3000, seed=42):
    """
    Relax positions using repulsive forces between circles and boundary constraints.
    """
    np.random.seed(seed)
    n = len(pos)
    pos = pos.copy()
    
    # Parameters
    k_repel = 1.0
    k_boundary = 5.0
    k_center = 0.01 # Weak attraction to center to keep packing compact
    step_size = 0.01
    cooling = 0.999
    
    for i in range(iterations):
        forces = np.zeros_like(pos)
        
        # 1. Pairwise Repulsion
        # Vectorized calculation of distances
        # This is O(N^2) memory if we do all pairs at once.
        # For N=300, 300x300 = 90k floats. This is fine.
        
        # Calculate pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Mask out self-interactions and very small distances to avoid div by zero
        mask = (dist > 1e-8) & (dist < 2.0 * r_target * 1.5)
        
        # Force magnitude: proportional to overlap / dist^2
        # F = k * (2r - d) / d^2 * unit_vec
        # unit_vec = diff / d
        # So F_vec = k * (2r - d) / d^3 * diff
        
        overlap = np.maximum(0, 2.0 * r_target - dist)
        
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-8, dist, 1.0)
        
        force_mag = np.where(mask, k_repel * overlap / (safe_dist**3), 0.0)
        
        # Apply forces
        # force_vec[i,j] = force_mag[i,j] * diff[i,j]
        # Sum over j for each i
        force_vec = force_mag[:, :, np.newaxis] * diff
        forces += np.sum(force_vec, axis=1)
        
        # 2. Boundary Constraint
        # If |p| > 1 - r, push towards center
        radii = np.hypot(pos[:, 0], pos[:, 1])
        boundary_violation = np.maximum(0, radii - (1.0 - r_target))
        
        # Force direction: towards center (-pos / |pos|)
        safe_radii = np.where(radii > 1e-8, radii, 1.0)
        boundary_force = -k_boundary * boundary_violation[:, np.newaxis] * pos / safe_radii[:, np.newaxis]
        forces += boundary_force
        
        # 3. Center Attraction (to prevent drift and maintain symmetry)
        forces -= k_center * pos
        
        # Update positions
        pos += step_size * forces
        
        # Cool down step size
        step_size *= cooling
        
        # Optional: Hard clamp to prevent flying out
        new_radii = np.hypot(pos[:, 0], pos[:, 1])
        max_allowed = 1.0
        mask_out = new_radii > max_allowed
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] * (max_allowed / new_radii[mask_out])[:, np.newaxis]
            
    return pos

def _compute_min_radius(positions):
    """Compute the maximum common radius for the given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Minimum distance to container boundary
    dist_to_boundary = 1.0 - np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = np.min(dist_to_boundary)
    
    if min_boundary < 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    # Use grid or brute force? For N=300, brute force vectorized is fast enough.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

# EVOLVE-BLOCK-END
