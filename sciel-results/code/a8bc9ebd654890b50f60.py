import numpy as np
import time

N = 200
TIME_LIMIT = 135.0  # Leave margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _fibonacci_lattice(n):
    """Generate a Fibonacci lattice (sunflower pattern) in the unit square."""
    # The golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Indices 1 to n
    i = np.arange(1, n + 1)
    
    # Standard sunflower formula:
    # r = sqrt(i / n)
    # theta = 2 * pi * i / phi
    # x = r * cos(theta)
    # y = r * sin(theta)
    
    # However, this gives a disk. We want a square.
    # A common way to map to a square is to use the area-preserving map or just
    # normalize the coordinates to fit the square [-0.5, 0.5].
    # The sunflower points are uniformly distributed in area.
    # The maximum radius in the disk is 1.
    # To fit in [-0.5, 0.5], we scale by 0.5.
    
    r = np.sqrt(i / n)
    theta = 2 * np.pi * i / phi
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit in [-0.5, 0.5]
    # The points are in a circle of radius 1.
    # The square [-0.5, 0.5] has half-diagonal sqrt(2)/2 ~ 0.707.
    # So we can scale by 0.5 to fit comfortably.
    # Actually, the sunflower points fill the circle of radius 1.
    # If we scale by 0.5, they fit in [-0.5, 0.5].
    x *= 0.5
    y *= 0.5
    
    return np.column_stack([x, y])

def _hexagonal_lattice(n):
    """Generate a hexagonal lattice configuration."""
    # Approximate number of rows
    # Density of hex packing is sqrt(3)/2 ~ 0.866
    # Area of square is 1.
    # Area per point is 1/n.
    # Hex cell area is s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 = 1/n => s = sqrt(2 / (n * sqrt(3)))
    s = np.sqrt(2 / (n * np.sqrt(3)))
    
    points = []
    # Determine range
    # y = j * s * sqrt(3) / 2
    max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
    for j in range(-max_j, max_j + 1):
        y = j * s * np.sqrt(3) / 2
        if abs(y) > 0.5:
            continue
        offset = (j % 2) * s / 2
        max_i = int(1.0 / s) + 2
        for i in range(-max_i, max_i + 1):
            x = i * s + offset
            if -0.5 <= x <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # We need exactly N points.
    # If we have more, we need to remove some.
    # If we have fewer, we need to add some or adjust s.
    # Let's try to adjust s to get close to N, then pick the best N.
    
    # For now, just return the first N if available, or pad if not.
    # A better approach: find the s that gives exactly N or closest.
    # But for initialization, a dense set is fine, we'll optimize.
    
    if len(points) < n:
        # Pad with random points? No, let's just use a grid if hex fails to produce enough.
        # This is unlikely for reasonable s.
        pass
    else:
        # Sort by distance from center? Or just take first N?
        # Taking first N from a loop is biased.
        # Let's pick the N points that are most "central" or just random N?
        # Random N might break symmetry.
        # Let's try to pick N points that are well-distributed.
        # Simple heuristic: sort by distance from origin and take N?
        # This would bias towards center.
        # Actually, for a lattice, just taking a contiguous block might be okay if we center it.
        
        # Let's just take the first N. The optimization will fix it.
        points = points[:n]
        
    return points

def _generate_initial_config():
    """Generate a high-quality initial configuration."""
    # Try both Fibonacci and Hexagonal, pick the one with better initial radius.
    fib_centres = _fibonacci_lattice(N)
    hex_centres = _hexagonal_lattice(N)
    
    r_fib = _compute_radius(fib_centres)
    r_hex = _compute_radius(hex_centres)
    
    if r_fib >= r_hex:
        return fib_centres
    else:
        return hex_centres

def _local_optimize(centres, time_limit):
    """
    Optimize the positions to maximize the minimum radius.
    Uses a repulsive force-based simulation (Langevin dynamics).
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    
    # Parameters
    temperature = 0.01
    min_temperature = 1e-6
    cooling_rate = 0.995
    force_cutoff = 0.1  # Only consider forces from neighbors within this distance? No, all.
    # Actually, for rigid constraints, we need to resolve all overlaps.
    # Let's use a simple force: F = k * (d - 2r) * (r_vec / d) for d < 2r.
    # But we don't know r exactly, we want to maximize it.
    # So we treat the current best_r as a target and push apart any pair closer than 2*best_r.
    # Also push away from walls if closer than best_r.
    
    # To make it smoother, we can use a soft potential.
    # V = 0.5 * k * max(0, 2r - d)^2
    # Force = k * (2r - d) * (r_vec / d)
    
    k = 10.0  # Spring constant
    dt = 0.001  # Time step
    damping = 0.9  # Damping factor
    
    velocities = np.zeros_like(best_centres)
    
    while time.time() < deadline and temperature > min_temperature:
        # Calculate forces
        forces = np.zeros_like(best_centres)
        
        # Pairwise forces
        # O(N^2) is fine for N=200
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = best_centres[i] - best_centres[j]
                d = np.linalg.norm(d_vec)
                if d < 2.0 * best_r:
                    # Overlap
                    if d > 1e-9:
                        force_mag = k * (2.0 * best_r - d)
                        force_dir = d_vec / d
                        forces[i] += force_mag * force_dir
                        forces[j] -= force_mag * force_dir
                    else:
                        # Coincident points, random push
                        force_dir = np.random.randn(2)
                        force_dir /= np.linalg.norm(force_dir)
                        forces[i] += force_mag * force_dir
                        forces[j] -= force_mag * force_dir
        
        # Wall forces
        # If point is too close to wall, push it in.
        # Distance to left wall: x + 0.5
        # If x + 0.5 < best_r, push right.
        for i in range(N):
            x, y = best_centres[i]
            
            # Left
            dist_left = x + 0.5
            if dist_left < best_r:
                forces[i, 0] += k * (best_r - dist_left)
            
            # Right
            dist_right = 0.5 - x
            if dist_right < best_r:
                forces[i, 0] -= k * (best_r - dist_right)
            
            # Down
            dist_down = y + 0.5
            if dist_down < best_r:
                forces[i, 1] += k * (best_r - dist_down)
            
            # Up
            dist_up = 0.5 - y
            if dist_up < best_r:
                forces[i, 1] -= k * (best_r - dist_up)
        
        # Update velocities and positions
        # v = v * damping + forces * dt + random_kick
        velocities = velocities * damping + forces * dt
        # Add thermal noise
        velocities += np.random.randn(N, 2) * temperature * 0.01
        
        new_centres = best_centres + velocities * dt
        
        # Enforce bounds strictly (clip)
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Evaluate
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            best_centres = new_centres
            # Cool down
            temperature *= cooling_rate
        else:
            # Revert? Or keep and hope?
            # In MD, we usually keep the state and let forces correct it.
            # But here we want to maximize min radius.
            # If new_r is worse, we might have introduced overlaps.
            # Let's keep the position but revert velocities?
            # Or just keep the position, the forces will push them apart next step.
            # However, if new_r is significantly worse, we might be stuck.
            # Let's keep the position for now, as it's a physical simulation.
            best_centres = new_centres
            # Don't cool down? Or cool slower?
            # temperature *= 0.999
        
    return best_centres, best_r

def _polish(centres, time_limit):
    """
    Final polish pass: local optimization with smaller steps to refine the solution.
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    
    step_size = 0.005
    min_step_size = 1e-5
    
    while time.time() < deadline and step_size > min_step_size:
        # Find bottleneck
        from scipy.spatial.distance import cdist
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        wall_clearances = np.min(np.stack([
            0.5 - np.abs(best_centres[:, 0]),
            0.5 - np.abs(best_centres[:, 1])
        ], axis=1), axis=1)
        
        min_neighbor_dists = np.min(dists, axis=1)
        clearances = np.minimum(min_neighbor_dists / 2.0, wall_clearances)
        
        bottleneck_idx = np.argmin(clearances)
        bottleneck_val = clearances[bottleneck_idx]
        
        if bottleneck_val > best_r + 1e-9:
            # We are not at the limit? This shouldn't happen if best_r is correct.
            # Maybe numerical error. Update best_r.
            best_r = bottleneck_val
            continue
        
        # Move bottleneck point
        cx, cy = best_centres[bottleneck_idx]
        
        # Determine direction
        # Check wall limits
        dist_left = cx + 0.5
        dist_right = 0.5 - cx
        dist_down = cy + 0.5
        dist_up = 0.5 - cy
        min_wall_dist = min(dist_left, dist_right, dist_down, dist_up)
        
        # Check neighbor limit
        neighbor_idx = np.argmin(dists[bottleneck_idx])
        min_neighbor_dist = dists[bottleneck_idx, neighbor_idx]
        
        if min_neighbor_dist / 2.0 < min_wall_dist:
            # Neighbor is constraint
            nx, ny = best_centres[neighbor_idx]
            dx = cx - nx
            dy = cy - ny
            norm = np.linalg.norm([dx, dy])
            if norm > 1e-9:
                dx /= norm
                dy /= norm
            else:
                dx, dy = 1.0, 0.0
        else:
            # Wall is constraint
            if min_wall_dist == dist_left:
                dx, dy = 1.0, 0.0
            elif min_wall_dist == dist_right:
                dx, dy = -1.0, 0.0
            elif min_wall_dist == dist_down:
                dx, dy = 0.0, 1.0
            else:
                dx, dy = 0.0, -1.0
        
        # Try moving in this direction
        new_centres = best_centres.copy()
        new_centres[bottleneck_idx] += [dx * step_size, dy * step_size]
        new_centres[bottleneck_idx] = np.clip(new_centres[bottleneck_idx], -0.5, 0.5)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            step_size *= 0.95
        else:
            step_size *= 0.9
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    centres = _generate_initial_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # 2. Local Optimization (Repulsive Force)
    # Allocate 80% of time to MD
    if time_budget > 10:
        md_time = time_budget * 0.8
        centres, current_r = _local_optimize(centres, md_time)
        
        # 3. Polish
        polish_time = time_budget * 0.2
        centres, current_r = _polish(centres, polish_time)
        
    # 4. Final Validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
