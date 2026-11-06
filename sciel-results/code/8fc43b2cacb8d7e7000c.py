"""Rewritten program: vectorized soft-constraint gradient descent for circle packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with a perturbed grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    centers += rng.normal(0, 0.001, size=centers.shape)
    
    # Target radius: aim for the best known value
    # The constraint is distance >= 2 * r
    r_target = 0.036612798904
    
    # Optimization parameters
    n_steps = 20000
    lr = 0.005
    momentum = np.zeros_like(centers)
    mu = 0.9
    
    for step in range(n_steps):
        # Decay learning rate slightly
        current_lr = lr / (1.0 + step * 0.0001)
        
        # Compute pairwise differences and distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        
        # Mask out diagonal (self-interaction)
        np.fill_diagonal(dist, np.inf)
        
        # Calculate forces
        # We want to penalize dist < 2 * r_target
        # Let gap = 2 * r_target - dist
        # Force is proportional to gap if gap > 0, else 0
        # Gradient of 0.5 * gap^2 is gap * d(gap)/d(centers)
        # gap = 2*r - dist
        # d(gap)/d(c_i) = - d(dist)/d(c_i) = - (c_i - c_j) / dist
        # So force on i from j = gap * (c_i - c_j) / dist
        # This pushes i away from j if gap > 0
        
        gap = 2 * r_target - dist
        # Only apply force if gap > 0
        mask = gap > 0
        # Avoid division by zero
        safe_dist = np.where(mask, dist, 1.0)
        
        # Force vector: diff * (gap / dist)
        # diff is (N, N, 2)
        # factor is (N, N)
        factor = np.zeros_like(dist)
        factor[mask] = gap[mask] / safe_dist[mask]
        
        # forces[i] = sum_j factor[i, j] * diff[i, j]
        forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Wall constraints
        # Centers must be in [-0.5 + r, 0.5 - r]
        # Let's use the same soft penalty for walls
        # x-wall: if x < -0.5 + r_target, push right. if x > 0.5 - r_target, push left.
        # Penalty: max(0, (-0.5 + r_target) - x)^2 + max(0, (0.5 - r_target) - x)^2 ?? No.
        # Distance to wall: d_wall = min(x - (-0.5), 0.5 - x)
        # We want d_wall >= r_target
        # gap_x_left = r_target - (x + 0.5)
        # gap_x_right = r_target - (0.5 - x)
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Left wall: x + 0.5 < r_target => x < r_target - 0.5
        gap_x_left = (r_target - 0.5) - x
        mask_x_left = gap_x_left > 0
        forces[:, 0] += np.where(mask_x_left, gap_x_left, 0)
        
        # Right wall: 0.5 - x < r_target => x > 0.5 - r_target
        gap_x_right = x - (0.5 - r_target)
        mask_x_right = gap_x_right > 0
        forces[:, 0] -= np.where(mask_x_right, gap_x_right, 0)
        
        # Bottom wall: y + 0.5 < r_target
        gap_y_bottom = (r_target - 0.5) - y
        mask_y_bottom = gap_y_bottom > 0
        forces[:, 1] += np.where(mask_y_bottom, gap_y_bottom, 0)
        
        # Top wall: 0.5 - y < r_target
        gap_y_top = y - (0.5 - r_target)
        mask_y_top = gap_y_top > 0
        forces[:, 1] -= np.where(mask_y_top, gap_y_top, 0)
        
        # Momentum update
        momentum = mu * momentum + current_lr * forces
        centers += momentum
        
        # Clip to valid region to prevent divergence (hard constraint for stability)
        centers = np.clip(centers, -0.5 + r_target * 0.9, 0.5 - r_target * 0.9)
        
    # Final verification and cleanup
    # The returned radius is determined by the evaluator.
    # We just return the centers.
    return centers
# EVOLVE-BLOCK-END
