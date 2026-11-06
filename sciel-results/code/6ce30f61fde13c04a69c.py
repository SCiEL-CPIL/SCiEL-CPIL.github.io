"""Rewritten program: fully vectorized soft-constraint gradient descent for circle packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with a hexagonal lattice pattern for better packing density
    # Hexagonal lattice has higher density than square lattice
    # For N=200, use a hex grid arrangement
    
    # Create hexagonal lattice points
    # Hex lattice: rows offset by half spacing
    spacing = 1.0 / np.sqrt(N) * 1.1  # slight expansion to fit in square
    
    # Generate hex lattice points
    hex_points = []
    row = 0
    while len(hex_points) < N:
        for col in range(int(1.2 / spacing) + 1):
            x = (col - (int(1.2 / spacing) + 1) / 2.0) * spacing
            y = (row - (int(1.2 / spacing) + 1) / 2.0) * spacing * 0.866  # sqrt(3)/2
            if row % 2 == 1:
                x += spacing / 2.0
            hex_points.append([x, y])
        row += 1
    
    centers = np.array(hex_points[:N])
    
    # Center the array
    centers -= centers.mean(axis=0)
    
    # Scale to fit in [-0.49, 0.49]
    max_extent = np.max(np.abs(centers))
    if max_extent > 0.48:
        centers *= 0.48 / max_extent
    
    # Add small random perturbation to break symmetry
    centers += rng.normal(0, 0.002, size=centers.shape)
    
    # Clip to be safely inside
    centers = np.clip(centers, -0.48, 0.48)
    
    # Target radius - we'll optimize this
    # Best known is ~0.036612798904
    r_target = 0.0365
    
    # Vectorized gradient descent with momentum
    n_steps = 8000
    lr_initial = 0.002
    lr_final = 0.00005
    momentum = 0.9
    
    velocity = np.zeros_like(centers)
    
    for step in range(n_steps):
        # Anneal learning rate
        progress = step / n_steps
        lr = lr_initial * (lr_final / lr_initial) ** progress
        
        # Compute all pairwise differences and distances (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)  # (N, N)
        
        # Mask out self-pairs
        mask = ~np.eye(N, dtype=bool)
        dist = np.where(mask, dist, np.inf)
        
        # Compute forces: for each pair i,j with d_ij < r_target,
        # force on i is proportional to (r_target - d_ij) in direction from j to i
        # force_ij = (r_target - d_ij) * (c_i - c_j) / d_ij if d_ij < r_target else 0
        
        # Vectorized force computation
        # weight_ij = max(0, r_target - d_ij) / d_ij
        violation = np.maximum(0, r_target - dist)  # (N, N)
        weight = violation / (dist + 1e-12)  # (N, N)
        weight = np.where(mask, weight, 0)
        
        # Force on center i = sum_j weight_ij * (c_i - c_j)
        # = sum_j weight_ij * diff[i, j]
        forces = np.sum(weight[:, :, np.newaxis] * diff, axis=1)  # (N, 2)
        
        # Wall forces: push centers away from walls
        # For each center and each coordinate, if too close to wall, add repulsive force
        wall_margin = r_target
        # Left wall (x = -0.5): if x < -0.5 + wall_margin, push right
        # Right wall (x = 0.5): if x > 0.5 - wall_margin, push left
        # Same for y
        
        wall_forces = np.zeros_like(centers)
        
        # x-coordinate walls
        left_viol = np.maximum(0, (-0.5 + wall_margin) - centers[:, 0])
        right_viol = np.maximum(0, centers[:, 0] - (0.5 - wall_margin))
        wall_forces[:, 0] += left_viol - right_viol
        
        # y-coordinate walls
        bottom_viol = np.maximum(0, (-0.5 + wall_margin) - centers[:, 1])
        top_viol = np.maximum(0, centers[:, 1] - (0.5 - wall_margin))
        wall_forces[:, 1] += bottom_viol - top_viol
        
        # Total force
        total_forces = forces + wall_forces
        
        # Update with momentum
        velocity = momentum * velocity + lr * total_forces
        centers += velocity
        
        # Clip to stay inside the square (with some margin)
        centers = np.clip(centers, -0.49, 0.49)
    
    # Final refinement with smaller steps
    for step in range(3000):
        progress = step / 3000
        lr = 0.0001 * (1 - progress) + 0.00001
        
        # Compute all pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        
        mask = ~np.eye(N, dtype=bool)
        dist = np.where(mask, dist, np.inf)
        
        violation = np.maximum(0, r_target - dist)
        weight = violation / (dist + 1e-12)
        weight = np.where(mask, weight, 0)
        
        forces = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        wall_margin = r_target
        wall_forces = np.zeros_like(centers)
        
        left_viol = np.maximum(0, (-0.5 + wall_margin) - centers[:, 0])
        right_viol = np.maximum(0, centers[:, 0] - (0.5 - wall_margin))
        wall_forces[:, 0] += left_viol - right_viol
        
        bottom_viol = np.maximum(0, (-0.5 + wall_margin) - centers[:, 1])
        top_viol = np.maximum(0, centers[:, 1] - (0.5 - wall_margin))
        wall_forces[:, 1] += bottom_viol - top_viol
        
        total_forces = forces + wall_forces
        
        velocity = 0.9 * velocity + lr * total_forces
        centers += velocity
        centers = np.clip(centers, -0.49, 0.49)
    
    # Verify and return
    return centers
# EVOLVE-BLOCK-END
