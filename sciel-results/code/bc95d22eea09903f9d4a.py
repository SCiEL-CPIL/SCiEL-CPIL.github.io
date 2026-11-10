"""Packing 200 equal circles in a unit square using Fibonacci lattice init + repulsive force refinement."""
import numpy as np
import time

N = 200

def _fibonacci_lattice(n):
    """Generate n points using the Fibonacci lattice (sunflower pattern) in [0,1)^2."""
    i = np.arange(n, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = i * golden_angle
    radii = np.sqrt((i + 0.5) / n)
    x = 0.5 + 0.5 * radii * np.cos(angles)
    y = 0.5 + 0.5 * radii * np.sin(angles)
    return np.column_stack([x - 0.5, y - 0.5])


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    # Distance between all pairs
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    min_boundary = np.min(np.min(0.5 - np.abs(centers), axis=0))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _repulsive_force_step(centers, r_target, steps=5000, lr=0.0005):
    """
    Refine centers using a repulsive force simulation.
    Each pair of circles repels if they overlap (distance < 2*r_target).
    Boundaries also repel if a circle goes outside.
    """
    n = len(centers)
    centers = centers.copy().astype(np.float64)
    
    for step in range(steps):
        # Compute pairwise distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive forces between overlapping circles
        # Force magnitude proportional to overlap amount
        overlap = np.maximum(0, 2 * r_target - dists)
        # Normalize direction
        mask = dists > 1e-10
        dirs = np.zeros_like(diffs)
        dirs[mask] = diffs[mask] / dists[mask][:, :, np.newaxis]
        
        # Force on each circle from all others
        force = np.sum(dirs * overlap[:, :, np.newaxis], axis=1)
        
        # Boundary forces
        # For each coordinate, if center is too close to boundary, push inward
        for dim in range(2):
            # Left boundary: x < -0.5 + r_target
            too_left = centers[:, dim] < -0.5 + r_target
            too_right = centers[:, dim] > 0.5 - r_target
            force[too_left, dim] += (r_target * 2 - (centers[too_left, dim] + 0.5)) * 0.5
            force[too_right, dim] -= (r_target * 2 - (0.5 - centers[too_right, dim])) * 0.5
        
        # Update positions
        centers += lr * force
        
        # Clamp to valid region (with small margin)
        margin = r_target * 0.1
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_target * 0.01, 0.5 - r_target * 0.01)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_target * 0.01, 0.5 - r_target * 0.01)
    
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial Fibonacci lattice
    centers = _fibonacci_lattice(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Try different target radii, starting from current best and increasing
    # The theoretical max is around 0.0366
    # Start with a slightly smaller target and increase
    
    # First, let's try to find a good radius through iterative refinement
    # Use the current best as a lower bound
    
    r_current = best_radius
    r_max_possible = 0.5 / np.sqrt(N)  # rough upper bound for grid-like packing
    
    # Binary search / iterative improvement on radius
    # We'll try to push the radius up by refining
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Try a slightly larger target radius
        r_target = r_current * 1.005 + 0.0001
        
        if r_target > r_max_possible:
            r_target = r_max_possible
        
        # Run repulsive force simulation with this target
        refined = _repulsive_force_step(centers, r_target, steps=2000, lr=0.0003)
        
        # Check if the refined configuration is valid and has better radius
        new_radius = _compute_radius(refined)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = refined.copy()
            centers = refined.copy()
            r_current = best_radius
        else:
            # If it didn't improve, keep the current centers but maybe adjust
            centers = refined.copy()
            # Don't update r_current, keep trying similar targets
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centers
