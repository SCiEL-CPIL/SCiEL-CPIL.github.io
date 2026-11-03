"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def compute_radius(coords):
    """
    Computes the maximum radius for the given circle centers.
    Returns the minimum of:
    1. Half the minimum distance between any two circle centers.
    2. The minimum distance from any circle center to the boundary of the square [-0.5, 0.5]^2.
    """
    if not np.all(np.isfinite(coords)):
        return 0.0
    
    # Distance to boundaries
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radius = min(min_wall_dist, min_pair_dist / 2.0)
    return radius

def initial_hex_layout(n):
    """
    Generates an initial hexagonal layout of n points inside the square.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)
    points = points[indices[:N]]
    return points

def force_directed_optimize(coords, max_iters=5000, start_temp=0.01, end_temp=0.001, cooling_rate=0.999):
    """
    Force-directed optimization with simulated annealing.
    Uses repulsive forces between circles and attractive forces toward center.
    """
    coords = coords.copy()
    best_radius = compute_radius(coords)
    best_coords = coords.copy()
    temp = start_temp
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(coords)
        
        # Repulsive forces between all pairs
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Force magnitude: inverse square law
        force_mag = 1.0 / (dists ** 2)
        np.fill_diagonal(force_mag, 0.0)
        
        # Sum forces from all other circles
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Attractive force toward center
        forces -= 0.1 * coords
        
        # Boundary forces (push back if too close to walls)
        dist_to_left = coords[:, 0] + 0.5
        dist_to_right = 0.5 - coords[:, 0]
        dist_to_bottom = coords[:, 1] + 0.5
        dist_to_top = 0.5 - coords[:, 1]
        
        boundary_force_x = np.zeros(N)
        boundary_force_y = np.zeros(N)
        
        # Left wall
        mask = dist_to_left < 0.05
        boundary_force_x[mask] += (0.05 - dist_to_left[mask]) * 100
        # Right wall
        mask = dist_to_right < 0.05
        boundary_force_x[mask] -= (0.05 - dist_to_right[mask]) * 100
        # Bottom wall
        mask = dist_to_bottom < 0.05
        boundary_force_y[mask] += (0.05 - dist_to_bottom[mask]) * 100
        # Top wall
        mask = dist_to_top < 0.05
        boundary_force_y[mask] -= (0.05 - dist_to_top[mask]) * 100
        
        forces[:, 0] += boundary_force_x
        forces[:, 1] += boundary_force_y
        
        # Add random perturbation (annealing)
        noise = np.random.normal(0, temp, coords.shape)
        forces += noise
        
        # Update positions
        step_size = 0.001
        coords += step_size * forces
        
        # Keep within bounds
        coords[:, 0] = np.clip(coords[:, 0], -0.4999, 0.4999)
        coords[:, 1] = np.clip(coords[:, 1], -0.4999, 0.4999)
        
        # Evaluate
        current_radius = compute_radius(coords)
        if current_radius > best_radius:
            best_radius = current_radius
            best_coords = coords.copy()
        
        # Cool down
        temp *= cooling_rate
        
        # Check time every 100 iterations
        if it % 100 == 0:
            if time.time() > time.time() - 140:  # Safety margin
                break
    
    return best_coords

def soft_min(x, k=100.0):
    """Soft minimum using log-sum-exp."""
    x = np.asarray(x)
    m = np.min(x)
    return m - np.log(np.sum(np.exp(k * (x - m)))) / k

def objective_soft(coords, k=100.0):
    """Soft objective for optimization."""
    if not np.all(np.isfinite(coords)):
        return 1e6
    
    dist_x = np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)
    wall_dists = np.concatenate([dist_x, dist_y])
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2)) / 2.0
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    all_constraints = np.concatenate([wall_dists, pair_dists])
    r_soft = soft_min(all_constraints, k)
    return -r_soft

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Initial guess: Hexagonal
    coords = initial_hex_layout(N)
    best_radius = compute_radius(coords)
    best_coords = coords.copy()
    
    # Step 1: Force-directed optimization with annealing
    if time.time() < deadline - 30:
        try:
            fd_coords = force_directed_optimize(coords, max_iters=5000, start_temp=0.01, end_temp=0.001)
            fd_radius = compute_radius(fd_coords)
            if fd_radius > best_radius:
                best_radius = fd_radius
                best_coords = fd_coords
        except Exception:
            pass
    
    # Step 2: Gradient-based polish using L-BFGS-B
    if time.time() < deadline - 10:
        try:
            x0 = best_coords.flatten()
            res = minimize(objective_soft, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() < deadline - 5:
                new_coords = res.x.reshape(N, 2)
                new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
                new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
                new_radius = compute_radius(new_coords)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_coords = new_coords
        except Exception:
            pass
    
    # Step 3: Final Powell polish
    if time.time() < deadline - 2:
        try:
            def objective_exact(c):
                return -compute_radius(c.reshape(N, 2))
            
            x0 = best_coords.flatten()
            res = minimize(objective_exact, x0, method='Powell',
                          options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12})
            
            if time.time() < deadline:
                new_coords = res.x.reshape(N, 2)
                new_coords[:, 0] = np.clip(new_coords[:, 0], -0.5, 0.5)
                new_coords[:, 1] = np.clip(new_coords[:, 1], -0.5, 0.5)
                new_radius = compute_radius(new_coords)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_coords = new_coords
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_coords)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_coords

# EVOLVE-BLOCK-END
