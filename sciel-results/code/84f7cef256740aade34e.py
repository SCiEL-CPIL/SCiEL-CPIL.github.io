import numpy as np
import time

N = 50

def compute_score(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check bounds
    # Points must be in [-0.5, 0.5]
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0

    # Compute pairwise distances
    # Using broadcasting for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Compute distance to walls
    # x distance to wall: 0.5 - abs(x)
    # y distance to wall: 0.5 - abs(y)
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _md_step(positions, velocities, dt, wall_limit, k_rep=100.0, damping=0.99):
    """
    Perform one MD step with velocity Verlet integration.
    
    Args:
        positions: (N, 2) array of particle positions
        velocities: (N, 2) array of particle velocities
        dt: timestep
        wall_limit: wall position (e.g., 0.5 for unit square centered at origin)
        k_rep: repulsion strength for overlapping particles
        damping: velocity damping factor
    
    Returns:
        new_positions, new_velocities
    """
    N = len(positions)
    forces = np.zeros_like(positions)
    
    # Compute pairwise repulsive forces (soft sphere model)
    # For each pair, if distance < 2*effective_radius, apply repulsive force
    # We use a soft potential to allow smooth dynamics
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Find the minimum distance to determine effective radius
    min_dist = np.min(dists)
    eff_radius = min_dist / 2.0
    
    # Apply repulsive forces for overlapping or near-overlapping pairs
    # Use a quadratic potential: F = k * (2*r_eff - d) if d < 2*r_eff
    threshold = 2.0 * eff_radius * 1.01  # small buffer
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < threshold and d > 1e-10:
                # Repulsive force magnitude
                overlap = threshold - d
                f_mag = k_rep * overlap
                # Force direction: from j to i (repel i from j)
                direction = diff[i, j] / d
                forces[i] += f_mag * direction
                forces[j] -= f_mag * direction
    
    # Apply wall repulsion
    for i in range(N):
        for dim in range(2):
            pos = positions[i, dim]
            if pos > wall_limit - eff_radius:
                # Near right/top wall
                penetration = pos - (wall_limit - eff_radius)
                if penetration > 0:
                    forces[i, dim] -= k_rep * penetration
            elif pos < -wall_limit + eff_radius:
                # Near left/bottom wall
                penetration = (-wall_limit + eff_radius) - pos
                if penetration > 0:
                    forces[i, dim] += k_rep * penetration
    
    # Velocity Verlet integration
    # v(t + dt/2) = v(t) + a(t) * dt/2
    accels = forces  # unit mass
    vel_half = velocities + accels * dt / 2.0
    
    # x(t + dt) = x(t) + v(t + dt/2) * dt
    new_positions = positions + vel_half * dt
    
    # Compute new forces at new positions
    forces_new = np.zeros_like(positions)
    diff_new = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
    dists_new = np.linalg.norm(diff_new, axis=2)
    np.fill_diagonal(dists_new, np.inf)
    
    min_dist_new = np.min(dists_new)
    eff_radius_new = min_dist_new / 2.0
    threshold_new = 2.0 * eff_radius_new * 1.01
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dists_new[i, j]
            if d < threshold_new and d > 1e-10:
                overlap = threshold_new - d
                f_mag = k_rep * overlap
                direction = diff_new[i, j] / d
                forces_new[i] += f_mag * direction
                forces_new[j] -= f_mag * direction
    
    # Wall forces at new positions
    for i in range(N):
        for dim in range(2):
            pos = new_positions[i, dim]
            if pos > wall_limit - eff_radius_new:
                penetration = pos - (wall_limit - eff_radius_new)
                if penetration > 0:
                    forces_new[i, dim] -= k_rep * penetration
            elif pos < -wall_limit + eff_radius_new:
                penetration = (-wall_limit + eff_radius_new) - pos
                if penetration > 0:
                    forces_new[i, dim] += k_rep * penetration
    
    # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    accels_new = forces_new
    new_velocities = vel_half + accels_new * dt / 2.0
    
    # Apply damping
    new_velocities *= damping
    
    return new_positions, new_velocities

def _expand_and_settle(positions, velocities, start_time, deadline, wall_start=0.6, wall_target=0.5, 
                        dt=0.001, k_rep=100.0, damping=0.99, expand_rate=0.0001):
    """
    Run MD simulation with slowly contracting walls to pack circles densely.
    
    Args:
        positions: initial (N, 2) positions
        velocities: initial (N, 2) velocities
        start_time: start timestamp
        deadline: deadline timestamp
        wall_start: initial wall position
        wall_target: target wall position (0.5 for unit square)
        dt: timestep
        k_rep: repulsion strength
        damping: velocity damping
        expand_rate: rate of wall contraction per step
    
    Returns:
        final_positions, final_velocities
    """
    wall = wall_start
    step = 0
    
    while time.time() < deadline - 5.0:
        # Contract walls slowly
        if wall > wall_target:
            wall = max(wall - expand_rate, wall_target)
        
        # Perform MD step
        positions, velocities = _md_step(positions, velocities, dt, wall, k_rep, damping)
        
        # Keep positions within current walls
        positions = np.clip(positions, -wall, wall)
        
        step += 1
        
        # Check time more frequently
        if step % 1000 == 0 and time.time() > deadline - 10.0:
            break
    
    # Final position: ensure within [-0.5, 0.5]
    positions = np.clip(positions, -0.5, 0.5)
    
    return positions, velocities

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115 seconds safety margin
    
    # Initial configuration: Jittered grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    points = grid.copy()
    # Add 50th point
    p50 = np.random.uniform(-0.4, 0.4, 2)
    points = np.vstack([points, p50])
    
    # Shuffle and jitter
    np.random.shuffle(points)
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, -0.45, 0.45)
    
    # Initialize velocities to zero
    velocities = np.zeros((N, 2))
    
    # Run MD with contracting walls
    # Start with walls slightly outside the target to allow relaxation
    wall_start = 0.55
    wall_target = 0.5
    
    # Phase 1: Relax with slightly larger walls
    positions, velocities = _expand_and_settle(
        points, velocities, start_time, deadline,
        wall_start=wall_start, wall_target=wall_start,
        dt=0.002, k_rep=50.0, damping=0.995, expand_rate=0
    )
    
    # Check time
    if time.time() > deadline - 20.0:
        positions = np.clip(positions, -0.5, 0.5)
        return positions
    
    # Phase 2: Slowly contract walls to target
    positions, velocities = _expand_and_settle(
        positions, velocities, start_time, deadline,
        wall_start=wall_start, wall_target=wall_target,
        dt=0.001, k_rep=100.0, damping=0.99, expand_rate=0.00005
    )
    
    # Check time
    if time.time() > deadline - 10.0:
        positions = np.clip(positions, -0.5, 0.5)
        return positions
    
    # Phase 3: Final settling with target walls
    positions, velocities = _expand_and_settle(
        positions, velocities, start_time, deadline,
        wall_start=wall_target, wall_target=wall_target,
        dt=0.0005, k_rep=200.0, damping=0.98, expand_rate=0
    )
    
    # Ensure final positions are within [-0.5, 0.5]
    positions = np.clip(positions, -0.5, 0.5)
    
    return positions
