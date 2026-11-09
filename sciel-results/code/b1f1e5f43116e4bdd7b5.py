"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # N ~ pi / (s^2 * sqrt(3)) => s ~ sqrt(pi / (N * sqrt(3)))
    s = np.sqrt(np.pi / (N * np.sqrt(3)))
    # Create grid
    k = int(1.0 / s) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                rows.append([x, y])
    
    points = np.array(rows)
    
    # If we have too many, select N closest to center? Or just take first N?
    # Taking closest to center is usually better for symmetry.
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback: Random points if lattice fails (unlikely)
        points = np.random.rand(N, 2) * 2.0 - 1.0
        # Ensure inside
        while np.any(np.hypot(points[:, 0], points[:, 1]) > 1.0):
            mask = np.hypot(points[:, 0], points[:, 1]) > 1.0
            points[mask] = np.random.rand(np.sum(mask), 2) * 2.0 - 1.0

    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    
    # Clip to unit circle (strictly inside)
    r_norm = np.hypot(points[:, 0], points[:, 1])
    mask = r_norm > 0.99
    points[mask] *= 0.99 / r_norm[mask, np.newaxis]

    velocities = np.zeros((N, 2))
    
    # Simulation Parameters
    dt = 0.02
    damping = 0.9
    k_rep = 10.0
    k_bound = 10.0
    
    # Dynamic radius control
    r_current = 0.01  # Start small
    best_r = 0.0
    best_points = points.copy()
    
    # Pre-allocate arrays for speed if possible, but N is small enough for dynamic
    
    while time.time() - start_time < deadline:
        # Calculate distances
        # diff[i, j] = points[i] - points[j]
        # This is O(N^2) memory if we do it all at once. 300x300x2 = 180,000 floats. ~1.4MB. Fine.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists[dists == 0] = 1e-6
        
        # Calculate Forces
        forces = np.zeros_like(points)
        
        # Repulsion
        # Only for pairs where dist < 2 * r_current
        # We want to push them apart if they are too close
        # Force magnitude proportional to overlap
        overlap = np.maximum(0, 2 * r_current - dists)
        # Force direction: from j to i is diff[i,j]
        # F_i = sum_j k * overlap * (p_i - p_j) / dist
        # Note: diff[i,j] is p_i - p_j
        # So force on i from j is k * overlap * diff[i,j] / dists[i,j]
        
        # Zero out diagonal
        np.fill_diagonal(overlap, 0)
        
        # Vectorized force calculation
        # factor[i,j] = k_rep * overlap[i,j] / dists[i,j]
        factor = k_rep * overlap / dists
        np.fill_diagonal(factor, 0)
        
        # forces[i] = sum_j factor[i,j] * diff[i,j]
        forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary Containment
        # If |p_i| > 1 - r_current, push in
        r_points = np.linalg.norm(points, axis=1)
        boundary_violation = np.maximum(0, r_points - (1 - r_current))
        # Force direction: -p_i / |p_i|
        # F_i = -k_bound * violation * p_i / |p_i|
        # Handle p_i = 0
        safe_r_points = np.where(r_points > 1e-6, r_points, 1.0)
        boundary_forces = -k_bound * boundary_violation[:, np.newaxis] * points / safe_r_points[:, np.newaxis]
        
        forces += boundary_forces
        
        # Update Velocities (Langevin)
        # v = v * damping + F * dt + noise
        # Noise amplitude can decay
        elapsed = time.time() - start_time
        noise_amp = 0.01 * np.exp(-elapsed / 50.0)
        noise = np.random.normal(0, noise_amp, points.shape)
        
        velocities = velocities * damping + forces * dt + noise
        points += velocities * dt
        
        # Check for explosion
        if not np.all(np.isfinite(points)):
            break
            
        # Calculate current max possible radius
        # min_dist is min off-diagonal dists
        min_dist = np.min(np.where(np.eye(N, dtype=bool), 1e9, dists))
        min_boundary = np.min(1 - r_points)
        
        current_max_r = min(min_dist, min_boundary) / 2.0
        
        if current_max_r > best_r:
            best_r = current_max_r
            best_points = points.copy()
            
        # Update r_current for next step's constraints
        # Grow slowly towards current_max_r
        # If current_max_r is much larger, we can grow faster
        # If current_max_r is smaller (due to noise/violation), we might need to shrink or stay
        # A safe strategy: r_current = max(r_current * 1.001, 0.01)
        # But we want to track the capability.
        # Let's try: r_current = 0.95 * current_max_r + 0.05 * r_current
        # This ensures r_current doesn't exceed what the current configuration supports by much,
        # but allows it to grow as the configuration improves.
        
        # However, if current_max_r drops due to noise, r_current drops too, which might be okay.
        # But we want to push for growth.
        # Alternative: Keep r_current monotonically increasing?
        # If we keep it monotonic, and the system fails, forces will be huge.
        # Let's use: r_target = current_max_r
        # r_current = r_current + 0.1 * (r_target - r_current)
        # This converges to r_target.
        
        r_current = r_current + 0.1 * (current_max_r - r_current)
        
        # Ensure r_current doesn't get too small
        if r_current < 0.01:
            r_current = 0.01
            
    # Final check: ensure best_points is valid
    # The score is computed by the evaluator.
    # We return best_points.
    
    return best_points

# EVOLVE-BLOCK-END
