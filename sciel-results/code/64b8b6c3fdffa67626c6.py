"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Packing
    # Hexagonal packing is denser than square packing, providing a better starting point.
    # We generate a hexagonal grid and select the N points closest to the center.
    
    # Approximate radius for hexagonal packing of N circles in unit square
    # Area per circle ~ (2r)^2 * sqrt(3)/2 = 2 * sqrt(3) * r^2
    # N * 2 * sqrt(3) * r^2 ~ 1 => r ~ 1 / sqrt(2 * sqrt(3) * N)
    # For N=200, r ~ 1 / sqrt(2 * 1.732 * 200) ~ 1 / sqrt(692) ~ 0.038
    # Let's generate a grid with spacing slightly smaller than ideal to ensure we have enough points
    r_est = 0.038
    spacing = 2 * r_est
    hex_spacing = spacing * np.sqrt(3) / 2
    
    # Generate hexagonal points
    points = []
    # We need to cover the square [-0.5, 0.5]
    # Number of rows/cols needed
    max_coord = 0.5 + spacing
    num_rows = int(2 * max_coord / hex_spacing) + 1
    num_cols = int(2 * max_coord / spacing) + 1
    
    for row in range(num_rows):
        y = -max_coord + row * hex_spacing
        offset = (row % 2) * (spacing / 2)
        for col in range(num_cols):
            x = -max_coord + col * spacing + offset
            points.append([x, y])
            
    points = np.array(points)
    
    # Keep only points inside the square (with a small margin to avoid edge issues initially)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # Sort by distance from center and pick the first N
    dist_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dist_to_center)
    positions = points[idx[:N]].copy()
    
    # Ensure we have exactly N points. If not, add random points or adjust.
    # With the generation above, we should have plenty.
    if len(positions) < N:
        # Fallback: add random points
        while len(positions) < N:
            p = np.random.uniform(-0.49, 0.49, size=2)
            positions = np.vstack([positions, [p]])
    else:
        positions = positions[:N]
        
    # Add tiny jitter to break symmetry
    positions += np.random.normal(0, 0.001, size=positions.shape)
    
    # 2. Physics-Based Relaxation
    # We want to maximize the minimum radius.
    # This is equivalent to minimizing the "overlap" or "penetration" if we assume a target radius.
    # However, a more direct approach is to treat it as a force-directed layout:
    # - Repulsive force between any two circles if they are closer than 2*r_target.
    # - Attractive force towards the center to keep them compact.
    # - Repulsive force from walls if they are closer than r_target.
    
    # Instead of fixing r, we can iterate:
    # 1. Start with a guess r.
    # 2. Run relaxation to find a configuration that fits r.
    # 3. If success, increase r. If fail, decrease r or refine.
    
    # A better, simpler approach for a single function call:
    # Use a gradient descent on a potential energy function that penalizes violations of a target radius.
    # But we don't know the optimal radius.
    
    # Alternative: Use the "Repulsive Force" model where the force magnitude depends on distance.
    # F_ij = k * (1/d_ij - 1/(2*r)) if d_ij < 2*r else 0.
    # This naturally pushes points apart until they touch.
    
    # Let's use a dynamic radius approach.
    # Start with a small radius and increase it, relaxing the system.
    
    r_current = 0.030
    r_target = 0.036 # Best known is ~0.0366
    
    # We will perform a sequence of relaxation steps, gradually increasing the target radius.
    # This helps escape local minima.
    
    positions = positions.copy()
    
    # Precompute indices for efficiency? N=200 is small. Vectorized ops are fine.
    
    # Relaxation Parameters
    alpha = 0.01 # Learning rate / step size
    iterations_per_stage = 200
    num_stages = 10
    
    # We'll try to push towards the best known radius
    r_min = 0.030
    r_max = 0.037
    
    for stage in range(num_stages):
        r_target = r_min + (r_max - r_min) * (stage / (num_stages - 1))
        
        # Check time
        if time.time() > deadline - 10:
            break
            
        for i in range(iterations_per_stage):
            if time.time() > deadline - 5:
                break
                
            forces = np.zeros_like(positions)
            
            # 1. Pairwise Repulsive Forces
            # diff[i, j] = pos[i] - pos[j]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            
            # We only care about pairs closer than 2 * r_target
            # To save time, we can mask, but for N=200, full matrix is fast.
            # dist = sqrt(dist_sq)
            # force_mag = k * (1/dist - 1/(2*r_target)) for dist < 2*r_target
            
            # Vectorized calculation
            # Avoid division by zero (diagonal is inf, so 1/inf = 0)
            dist = np.sqrt(dist_sq)
            
            # Threshold
            threshold = 2 * r_target
            
            # Calculate force magnitude
            # If dist < threshold, force is repulsive (positive)
            # Force direction is along diff
            # F_ij = diff_ij * (1/dist_ij - 1/threshold) / dist_ij ? 
            # Actually, potential V = 1/d. Force = -grad V = 1/d^2 * r_hat = diff / d^3.
            # We want a force that vanishes at d = threshold.
            # Let's use a soft repulsion: F = k * (threshold - dist) / dist * r_hat
            # This is linear spring-like repulsion when overlapping.
            
            # Mask for overlapping pairs
            mask = dist < threshold
            # For non-overlapping, force is 0.
            
            # Compute force contribution
            # F_ij = diff_ij * (threshold - dist_ij) / dist_ij^2
            # Note: (threshold - dist) is positive if overlapping.
            # (1/dist) is the direction magnitude? No.
            # r_hat = diff / dist.
            # Magnitude = k * (threshold - dist).
            # So F = diff / dist * (threshold - dist) = diff * (threshold - dist) / dist.
            
            # To avoid division by zero for non-overlapping (where we don't care), 
            # we can compute only for mask.
            # But vectorized is faster.
            # Let's compute term = (threshold - dist) / dist
            # Where dist > 0.
            
            term = np.zeros_like(dist)
            valid = dist > 1e-6
            term[valid] = (threshold - dist[valid]) / dist[valid]
            
            # Force is diff * term
            # But we only want repulsion if overlapping.
            # If dist > threshold, (threshold - dist) is negative, which would be attractive.
            # We want 0 force if not overlapping.
            # So we mask the term to be 0 if dist >= threshold.
            term[~mask] = 0
            
            forces += np.sum(diff * term[:, :, np.newaxis], axis=1)
            
            # 2. Wall Repulsion
            # Distance to walls
            # x walls: -0.5, 0.5. y walls: -0.5, 0.5.
            # Force from left wall: if x < -0.5 + r_target, push right.
            # d_left = x - (-0.5) = x + 0.5
            # If d_left < r_target, force_x += k_wall * (r_target - d_left)
            
            x, y = positions[:, 0], positions[:, 1]
            
            # Left Wall
            d_left = x + 0.5
            mask_left = d_left < r_target
            forces[mask_left, 0] += (r_target - d_left[mask_left])
            
            # Right Wall
            d_right = 0.5 - x
            mask_right = d_right < r_target
            forces[mask_right, 0] -= (r_target - d_right[mask_right])
            
            # Bottom Wall
            d_bottom = y + 0.5
            mask_bottom = d_bottom < r_target
            forces[mask_bottom, 1] += (r_target - d_bottom[mask_bottom])
            
            # Top Wall
            d_top = 0.5 - y
            mask_top = d_top < r_target
            forces[mask_top, 1] -= (r_target - d_top[mask_top])
            
            # 3. Center Attraction (Weak)
            # To prevent the system from drifting to the edges and forming a ring,
            # we add a small attractive force to the center.
            forces -= 0.01 * positions
            
            # Update positions
            positions += alpha * forces
            
            # Clamp to square (hard constraint)
            positions = np.clip(positions, -0.5, 0.5)
            
            # Optional: Project onto feasible set? 
            # The forces should handle it, but clamping ensures validity.
            
    # Final refinement with a smaller step size and the best known radius target
    # Try to push slightly above 0.0366 if possible
    
    r_final_target = 0.036612798904
    
    # Run a few more iterations with this specific target
    alpha_refine = 0.005
    for i in range(500):
        if time.time() > deadline - 2:
            break
            
        forces = np.zeros_like(positions)
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        threshold = 2 * r_final_target
        mask = dist < threshold
        
        term = np.zeros_like(dist)
        valid = dist > 1e-6
        term[valid] = (threshold - dist[valid]) / dist[valid]
        term[~mask] = 0
        
        forces += np.sum(diff * term[:, :, np.newaxis], axis=1)
        
        x, y = positions[:, 0], positions[:, 1]
        
        d_left = x + 0.5
        mask_left = d_left < r_final_target
        forces[mask_left, 0] += (r_final_target - d_left[mask_left])
        
        d_right = 0.5 - x
        mask_right = d_right < r_final_target
        forces[mask_right, 0] -= (r_final_target - d_right[mask_right])
        
        d_bottom = y + 0.5
        mask_bottom = d_bottom < r_final_target
        forces[mask_bottom, 1] += (r_final_target - d_bottom[mask_bottom])
        
        d_top = 0.5 - y
        mask_top = d_top < r_final_target
        forces[mask_top, 1] -= (r_final_target - d_top[mask_top])
        
        forces -= 0.01 * positions
        
        positions += alpha_refine * forces
        positions = np.clip(positions, -0.5, 0.5)
        
    return positions

# EVOLVE-BLOCK-END
