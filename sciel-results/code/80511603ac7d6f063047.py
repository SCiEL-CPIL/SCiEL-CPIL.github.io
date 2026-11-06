"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum
    pairwise distance and distance to boundary.
    
    Strategy:
    1. Initialize using Fibonacci sphere distribution mapped to the disk.
    2. Apply iterative repulsion-based relaxation with soft penalty functions
       for overlaps and boundary violations.
    """
    rng = np.random.default_rng(42)
    
    # --- Step 1: Initialize using Fibonacci sphere mapped to disk ---
    # Fibonacci sphere gives uniform distribution on sphere; project to disk
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    
    # Fibonacci sphere: uniform points on sphere
    # z from -1 to 1 uniformly, theta golden angle
    z = 1.0 - 2.0 * (indices + 0.5) / n
    theta = golden_angle * indices
    r_sphere = np.sqrt(1.0 - z**2)
    x_sphere = r_sphere * np.cos(theta)
    y_sphere = r_sphere * np.sin(theta)
    
    # Map sphere to disk: use stereographic-like projection or simple scaling
    # For uniform area distribution in disk, use r_disk = sqrt((1 + z)/2) 
    # This maps z=-1 -> r=0, z=1 -> r=1, with uniform area density
    r_disk = np.sqrt((1.0 + z) / 2.0)
    x = r_disk * np.cos(theta)
    y = r_disk * np.sin(theta)
    
    centers = np.column_stack([x, y])
    
    # Ensure all points are strictly inside the unit disk
    radii = np.linalg.norm(centers, axis=1)
    mask = radii < 1.0
    if not np.all(mask):
        centers = centers[mask]
        if len(centers) < N:
            extra_n = N - len(centers)
            extra_r = np.sqrt(rng.random(extra_n)) * 0.95
            extra_theta = rng.random(extra_n) * 2 * np.pi
            extra_x = extra_r * np.cos(extra_theta)
            extra_y = extra_r * np.sin(extra_theta)
            extra_centers = np.column_stack([extra_x, extra_y])
            centers = np.vstack([centers, extra_centers])
        centers = centers[:N]
    
    # --- Step 2: Repulsion-based relaxation with soft penalties ---
    # We maximize the minimum of (pairwise distances, boundary distances)
    # Using a soft penalty: for each constraint, penalize violations smoothly
    
    n_iterations = 8000
    initial_step = 0.05
    min_step = 1e-10
    decay = 0.9997
    momentum = 0.3
    
    # Velocity for momentum
    velocity = np.zeros_like(centers)
    
    step_size = initial_step
    best_centers = centers.copy()
    best_score = -np.inf
    
    for it in range(n_iterations):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Set diagonal to large value
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Compute repulsion forces
        # Use a force that is strong for close pairs, weak for far pairs
        # F_ij = (1 / dist^3) * (c_i - c_j) for close pairs
        # To avoid numerical issues, clamp minimum distance
        dist_clamped = np.maximum(dist, 1e-6)
        dist_sq_clamped = dist_clamped ** 2
        
        # Force magnitude: 1/dist^2, direction: (c_i - c_j)/dist
        # So force = (c_i - c_j) / dist^3
        force_mag = 1.0 / (dist_sq_clamped * dist_clamped)
        np.fill_diagonal(force_mag, 0.0)
        
        # Only apply strong repulsion to pairs that are too close
        # This focuses computation on the critical pairs
        threshold = 0.3  # Only strongly repel pairs closer than this
        close_mask = dist < threshold
        force_mag[~close_mask] = 0.0
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force: smooth penalty for being close to boundary
        radii = np.linalg.norm(centers, axis=1)
        boundary_margin = 0.05
        # Smooth boundary force: quadratic penalty when r > 1 - margin
        boundary_forces = np.zeros_like(centers)
        excess = np.maximum(radii - (1.0 - boundary_margin), 0.0)
        # Force direction: inward
        inward = -centers / np.maximum(radii, 1e-6)[:, np.newaxis]
        boundary_forces = inward * (excess * 20.0)[:, np.newaxis]
        
        total_forces = forces + boundary_forces
        
        # Update with momentum
        velocity = momentum * velocity + (1.0 - momentum) * total_forces
        centers = centers + step_size * velocity
        
        # Keep points inside the unit disk (hard constraint)
        radii = np.linalg.norm(centers, axis=1)
        outside = radii >= 0.999
        if np.any(outside):
            centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.999
            velocity[outside] = 0.0  # Reset velocity for points that hit boundary
        
        # Compute current score
        dist_new = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair_dist = np.min(dist_new)
        min_boundary_dist = 1.0 - np.max(np.linalg.norm(centers, axis=1))
        current_score = min(min_pair_dist, min_boundary_dist)
        
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
        
        # Decay step size
        step_size *= decay
        if step_size < min_step:
            break
    
    # Final cleanup: ensure all points are strictly inside
    radii = np.linalg.norm(best_centers, axis=1)
    outside = radii >= 1.0
    if np.any(outside):
        best_centers[outside] = best_centers[outside] / radii[outside, np.newaxis] * 0.999
    
    return best_centers
# EVOLVE-BLOCK-END
