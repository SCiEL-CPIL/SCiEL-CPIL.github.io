"""Pack 50 equal circles in a unit square, maximize common radius.
Strategy: Hard-constraint repulsion force model with adaptive step size and momentum.
Avoids soft-min surrogate bias; directly optimizes true radius.
Uses hexagonal lattice initialization with perturbation.
"""
import numpy as np
import time

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _true_radius(centers):
    """Compute the exact maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius = min of (0.5 - |coord|)
    # If any center is outside, radius is 0 (or negative, but we clamp)
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r <= 0:
        return 0.0
    # Pairwise distances
    # Vectorized pairwise distance calculation
    # dist[i, j] = |c_i - c_j|
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _compute_forces(centers):
    """
    Compute repulsive forces to increase the minimum radius.
    Force on each circle is the gradient of the 'soft' minimum radius,
    but we use a hard repulsion from the current bottleneck constraints.
    
    Actually, a simpler and more robust method for this specific problem:
    Identify the critical pairs (those defining the current minimum radius)
    and the critical walls. Push circles apart from critical neighbors
    and away from critical walls.
    """
    r = _true_radius(centers)
    if r <= 0:
        return np.zeros_like(centers)
    
    forces = np.zeros_like(centers)
    
    # 1. Wall forces
    # If a circle is close to a wall, push it inward
    for i in range(N):
        for dim in range(2):
            # Check left wall (x < 0)
            dist_left = centers[i, dim] - (-0.5)
            if dist_left < r + 1e-9:
                # Push right
                forces[i, dim] += 1.0
            # Check right wall (x > 0)
            dist_right = 0.5 - centers[i, dim]
            if dist_right < r + 1e-9:
                # Push left
                forces[i, dim] -= 1.0
    
    # 2. Pairwise forces
    # We need to push circles that are at distance 2r apart further apart.
    # To avoid O(N^2) Python loop overhead, we can vectorize the check,
    # but applying forces requires iteration or careful indexing.
    # Given N=50, O(N^2) = 2500 checks is fast in Python if not in a tight inner loop with heavy ops.
    # However, let's try to vectorize the force calculation as much as possible.
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Mask for pairs that are too close (dist < 2r)
    # We want to push them apart.
    # Force direction on i from j is (c_i - c_j) / |c_i - c_j|
    # If dist < 2r, we apply a repulsive force.
    
    # To be efficient, we can identify indices where dist < 2r
    # and accumulate forces.
    
    # Since N is small, a simple loop over pairs is acceptable and less error-prone.
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 2 * r:
                # They are overlapping or touching. Push apart.
                if d < 1e-12:
                    # Random push if coincident
                    forces[i, 0] += np.random.uniform(-1, 1)
                    forces[i, 1] += np.random.uniform(-1, 1)
                    forces[j, 0] -= np.random.uniform(-1, 1)
                    forces[j, 1] -= np.random.uniform(-1, 1)
                else:
                    # Unit vector from j to i
                    u = diff[i, j] / d
                    # Apply force
                    forces[i] += u
                    forces[j] -= u

    return forces

def _local_refine_force(centers, time_limit=10.0):
    """
    Refine configuration using steepest ascent on true radius with adaptive step size.
    """
    x = centers.copy()
    best_x = x.copy()
    best_r = _true_radius(x)
    
    start_time = time.time()
    step_size = 0.01
    momentum = 0.0
    vel = np.zeros_like(x)
    
    while time.time() - start_time < time_limit:
        r = _true_radius(x)
        if r > best_r:
            best_r = r
            best_x = x.copy()
        
        # Compute forces
        forces = _compute_forces(x)
        
        # Normalize forces to prevent explosion
        norm_f = np.linalg.norm(forces)
        if norm_f > 1e-12:
            forces = forces / norm_f
        
        # Update velocity with momentum
        vel = 0.9 * vel + 0.1 * forces
        
        # Adaptive step size
        # Try a step
        new_x = x + step_size * vel
        
        # Check validity (must be inside square)
        # If any center goes outside, reject step and reduce step size
        if np.any(new_x < -0.5) or np.any(new_x > 0.5):
            step_size *= 0.5
            continue
            
        new_r = _true_radius(new_x)
        
        if new_r > r:
            x = new_x
            # Increase step size slightly if improving
            step_size = min(step_size * 1.1, 0.05)
        else:
            # Decrease step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
                
    return best_x

def _make_hex_init(perturb=0.0):
    """
    Generate a hexagonal lattice initialization.
    Hexagonal packing is denser than square.
    For 50 circles, a 7x8 rectangular grid is close, but hex is better.
    Let's try a hex grid with appropriate spacing.
    """
    # Estimate spacing for 50 points in unit square.
    # Area per point = 1/50 = 0.02.
    # Hex cell area = s^2 * sqrt(3)/2.
    # s^2 * 0.866 = 0.02 => s^2 = 0.023 => s = 0.15.
    s = 0.15
    dy = s * np.sqrt(3) / 2
    
    points = []
    # We need to fit 50 points.
    # Try a grid of 8x7 or similar.
    # Let's generate a large hex grid and pick the best 50 centered.
    
    # Generate grid covering [-0.6, 0.6]
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s
            y = j * dy
            if j % 2 != 0:
                x += s / 2
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter points roughly inside the square
    mask = (points[:, 0] > -0.5) & (points[:, 0] < 0.5) & (points[:, 1] > -0.5) & (points[:, 1] < 0.5)
    points = points[mask]
    
    if len(points) < N:
        # Fallback to random if not enough points
        return np.random.uniform(-0.45, 0.45, size=(N, 2))
        
    # Sort by distance from center to pick the most central ones?
    # Or just take the first N?
    # Better: pick the N points that maximize the minimum distance to the boundary?
    # Simple heuristic: take the N points closest to the origin?
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    init = points[idx]
    
    if perturb > 0:
        init += np.random.uniform(-perturb, perturb, size=init.shape)
        
    # Ensure inside square
    init = np.clip(init, -0.49, 0.49)
    
    return init

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Hexagonal initialization with various perturbations
    # The hex lattice is a strong prior for dense packing.
    
    np.random.seed(42)
    
    # Try a few different hex-based inits
    inits = []
    
    # 1. Pure hex
    c_hex = _make_hex_init(perturb=0.0)
    inits.append(c_hex)
    
    # 2. Perturbed hex
    for p in [0.01, 0.02, 0.05]:
        np.random.seed(42 + int(p*1000))
        c_hex_p = _make_hex_init(perturb=p)
        inits.append(c_hex_p)
        
    # 3. Random restarts from hex
    for _ in range(3):
        np.random.seed(int(time.time() * 1000) % 100000)
        c_hex_r = _make_hex_init(perturb=0.03)
        inits.append(c_hex_r)
        
    # 4. Square grid (fallback/comparison)
    # 7x8 grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    inits.append(grid)
    
    # Refine each init
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        time_left = DEADLINE - 20 - (time.time() - start_time)
        alloc_time = min(15.0, time_left / max(1, len(inits) - i))
        
        refined = _local_refine_force(init, time_limit=alloc_time)
        r = _true_radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # Phase 2: Polish the best candidate
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        # Use a longer time limit for polishing
        refined_polish = _local_refine_force(best_centers, time_limit=time_left * 0.8)
        r_polish = _true_radius(refined_polish)
        
        if r_polish > best_r:
            best_r = r_polish
            best_centers = refined_polish.copy()
            
    # Phase 3: If time remains, try more random restarts
    while time.time() - start_time < DEADLINE - 3:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 3 - (time.time() - start_time)
        if time_left < 3: break
        
        refined = _local_refine_force(c_rand, time_limit=time_left * 0.9)
        r = _true_radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    return best_centers
