"""
Pack 50 equal circles in a unit square using Physics-Based Hard-Disk Expansion.
Strategy:
1. Initialize with diverse configurations (grid, hex, random).
2. Use Molecular Dynamics (Velocity Verlet) with a repulsive potential 
   that becomes very stiff as circles approach contact.
3. Implement a cooling schedule to reduce velocity and allow settling.
4. Maximize the minimum radius by expanding the system or adjusting 
   the target radius until the system can no longer accommodate it.
"""

import numpy as np
import time

N = 50
DEADLINE = 145.0  # Safety margin

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _physics_refine(centers, max_time=10.0, r_target=None):
    """
    Refine centers using Molecular Dynamics.
    If r_target is provided, we try to find a configuration where all circles 
    of radius r_target fit. Otherwise, we just minimize overlap/violations 
    to find a locally dense packing, then calculate the actual radius.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    velocities = np.zeros_like(centers)
    
    # Parameters
    dt = 0.001
    k_rep = 1000.0  # Repulsion strength
    damping = 0.98  # Velocity damping (cooling)
    wall_k = 5000.0 # Wall repulsion strength (stronger to keep inside)
    
    # If we have a target radius, we use a potential that penalizes overlap 
    # relative to that radius.
    # If not, we just minimize the "soft" overlap to find a packing, 
    # but the goal is to maximize r. 
    # A better approach: Start with a small r, expand it.
    # Here, we assume the input centers are already "feasible" for some r.
    # We will run MD to minimize the potential energy defined by overlaps 
    # at the *current* feasible radius, or slightly larger.
    
    # Let's determine the current radius
    current_r = _radius(centers)
    
    # We want to increase r. 
    # Strategy: 
    # 1. Set r_sim = current_r * 1.01 (try to grow slightly)
    # 2. Run MD with potential based on r_sim.
    # 3. If the system stabilizes with low energy (low violation), accept new r.
    # 4. Repeat.
    
    # To be efficient, we will run a single long MD simulation 
    # that dynamically adjusts the "target" radius based on the current 
    # minimum distance, trying to push it up.
    
    # However, a simpler and robust method is:
    # Run MD to minimize the "soft" overlap potential for a fixed r_target.
    # If we succeed (energy ~ 0), we have a valid packing for r_target.
    # 
    # Let's implement a function that tries to fit circles of radius r_target.
    
    def _potential_and_force(x, r_t):
        """
        Compute potential energy and forces for hard disks of radius r_t.
        Uses a stiff repulsive potential for overlaps.
        """
        c = x.reshape(N, 2)
        forces = np.zeros_like(c)
        energy = 0.0
        
        # Pairwise interactions
        # Vectorized calculation for speed
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Overlap condition: dist < 2 * r_t
        # We use a potential U = 0.5 * k * (2*r_t - dist)^2 if dist < 2*r_t
        # To avoid division by zero or issues, we mask.
        
        mask = dist < 2 * r_t
        if np.any(mask):
            # Force magnitude: k * (2*r_t - dist)
            # Direction: diff / dist
            # We need to handle dist=0 (coincident points) by adding small noise
            # But in MD, we start from non-coincident points usually.
            
            # Create a matrix of forces
            # F_ij = -dU/dx_ij = k * (2*r_t - dist) * (diff / dist)
            # Note: diff is c_i - c_j. Force on i is in direction of diff.
            
            # Avoid division by zero
            safe_dist = np.where(mask, dist, 1.0)
            # Only calculate where mask is True to save time? 
            # Vectorized is fine for N=50.
            
            overlap = 2 * r_t - dist
            # Zero out non-overlaps
            overlap[~mask] = 0.0
            
            # Force factor: k * overlap / dist
            # Force on i: sum_j k * overlap_ij * (c_i - c_j) / dist_ij
            # This is equivalent to: 
            # F_i = k * sum_j (overlap_ij / dist_ij) * (c_i - c_j)
            
            # Let's compute the term (overlap / dist)
            # dist is 50x50. overlap is 50x50.
            # term = overlap / safe_dist
            # term[~mask] = 0
            
            term = np.zeros_like(dist)
            term[mask] = overlap[mask] / safe_dist[mask]
            
            # Force on i is sum_j term[i,j] * (c_i - c_j)
            # c_i - c_j is diff[i,j]
            # F_i = sum_j term[i,j] * diff[i,j]
            forces += np.sum(term[:, :, np.newaxis] * diff, axis=1) * k_rep
            
            # Energy
            energy += 0.5 * k_rep * np.sum(overlap[mask] ** 2)
        
        # Wall interactions
        # Push in if c + r_t > 0.5 or c - r_t < -0.5
        # x-wall:
        # Right: c_x + r_t > 0.5 -> dist_wall = c_x + r_t - 0.5
        # Left: c_x - r_t < -0.5 -> dist_wall = c_x - r_t + 0.5
        
        # Right wall
        d_right = c[:, 0] + r_t - 0.5
        mask_right = d_right > 0
        if np.any(mask_right):
            f_mag = wall_k * d_right[mask_right]
            forces[mask_right, 0] -= f_mag # Push left (negative x)
            energy += 0.5 * wall_k * np.sum(d_right[mask_right] ** 2)
            
        # Left wall
        d_left = c[:, 0] - r_t + 0.5
        mask_left = d_left < 0
        if np.any(mask_left):
            f_mag = wall_k * (-d_left[mask_left])
            forces[mask_left, 0] += f_mag # Push right (positive x)
            energy += 0.5 * wall_k * np.sum(d_left[mask_left] ** 2)
            
        # Top wall
        d_top = c[:, 1] + r_t - 0.5
        mask_top = d_top > 0
        if np.any(mask_top):
            f_mag = wall_k * d_top[mask_top]
            forces[mask_top, 1] -= f_mag # Push down
            energy += 0.5 * wall_k * np.sum(d_top[mask_top] ** 2)
            
        # Bottom wall
        d_bot = c[:, 1] - r_t + 0.5
        mask_bot = d_bot < 0
        if np.any(mask_bot):
            f_mag = wall_k * (-d_bot[mask_bot])
            forces[mask_bot, 1] += f_mag # Push up
            energy += 0.5 * wall_k * np.sum(d_bot[mask_bot] ** 2)
            
        return energy, forces

    # Main MD Loop
    # We will try to increase r from current_r up to a limit.
    # We start with r_sim = current_r.
    # We run MD to settle.
    # Then we try r_sim * 1.005.
    
    r_sim = current_r
    # If we are very close to the boundary, we might not be able to grow.
    # Let's set a max r based on area or just a reasonable upper bound like 0.08.
    max_possible_r = 0.08 
    
    # If current_r is already high, we just polish.
    # We will iterate:
    # 1. Run MD for a fixed number of steps to minimize energy at r_sim.
    # 2. If energy is low enough (violation < tol), try to increase r_sim.
    # 3. If energy is high, reduce r_sim or accept and stop.
    
    # To save time, we don't do binary search. We just do a "push" loop.
    
    # Initial step: settle at current r
    # We run a short MD to ensure it's stable at current r.
    
    n_steps_per_check = 500
    max_checks = 20
    
    for check_idx in range(max_checks):
        if time.time() - start_time > max_time:
            break
        
        # Run MD for n_steps
        for step in range(n_steps_per_check):
            # Velocity Verlet
            # 1. Update positions
            centers += velocities * dt + 0.5 * forces * dt ** 2
            
            # 2. Compute new forces
            energy, new_forces = _potential_and_force(centers, r_sim)
            
            # 3. Update velocities (with damping)
            velocities = (velocities + 0.5 * (forces + new_forces) * dt) * damping
            forces = new_forces
            
            # Check time
            if time.time() - start_time > max_time:
                break
        
        # After settling, check if we can increase r
        # We check the actual radius of the current configuration
        actual_r = _radius(centers)
        
        # If actual_r is significantly larger than r_sim, we can jump r_sim up.
        # If actual_r is close to r_sim, we try a small increment.
        
        # We want to maximize actual_r.
        # If the system is stable (energy low), we can try to increase r_sim.
        # How to measure stability? Energy should be close to 0.
        # Let's just try to increase r_sim by a small factor.
        
        # If we hit a wall (energy high), we might need to cool more or give up.
        # For simplicity, we just increase r_sim and let MD handle it.
        # If MD fails to reduce energy, the next _radius call will reflect the violation.
        
        # Increment r_sim
        r_sim = min(r_sim * 1.005, max_possible_r)
        
        # If we are not making progress, stop?
        # Hard to tell without tracking history.
        # We will just run until time limit or max_checks.
        
    # Final check
    final_r = _radius(centers)
    return centers, final_r

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def construct():
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8
    inits.append(_make_grid_rect(7, 8))
    
    # 2. Rectangular Grid 8x7
    inits.append(_make_grid_rect(8, 7))
    
    # 3. Hexagonal Grids
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 5. Another random with different seed
    np.random.seed(123)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Process each init
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        
        # Allocate time for physics refinement
        # Physics is fast, but we want to give it enough time to converge.
        # 10s per init is reasonable.
        alloc_time = min(10.0, time_left)
        
        if alloc_time < 1.0:
            continue
            
        refined_centers, r_val = _physics_refine(init, max_time=alloc_time)
        
        # Verify
        r_val = _radius(refined_centers)
        
        if r_val > best_r:
            best_r = r_val
            best_centers = refined_centers.copy()
            
    # If time remains, try more random restarts
    while time.time() - start_time < DEADLINE - 2:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 2 - (time.time() - start_time)
        if time_left < 1.0:
            break
            
        alloc_time = min(5.0, time_left)
        refined_centers, r_val = _physics_refine(c_rand, max_time=alloc_time)
        r_val = _radius(refined_centers)
        
        if r_val > best_r:
            best_r = r_val
            best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback to grid
        best_centers = _make_grid_rect(7, 8)
        
    return best_centers
