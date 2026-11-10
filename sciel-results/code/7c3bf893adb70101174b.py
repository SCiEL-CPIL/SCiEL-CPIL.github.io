import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialize with Hexagonal Lattice
    # Estimate spacing. Area of unit circle is pi. Area per circle is approx pi * r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N)
    # For hex packing, density is pi/sqrt(12) * (1/r_spacing)^2 ... 
    # Spacing s = 2r.
    # Let's just generate a dense grid and pick the best N.
    
    # Generate hex grid points
    # s is the distance between centers.
    # We want to cover the unit disk.
    # Max radius is approx 1/sqrt(300) * sqrt(2/3) ~ 0.054
    # So spacing is approx 0.1
    # Let's start with a spacing that fits ~300 points.
    
    points = _generate_hex_lattice(N)
    
    # 2. Local Optimization
    deadline = time.time() + 110.0 # Safety margin
    
    # Compute initial score
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Optimization loop
    # We use a force-based relaxation.
    # r_target is the radius we are trying to achieve.
    r_target = best_r
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try to increase r_target slightly
        # The improvement factor should be small to ensure convergence
        r_target = best_r * (1.0 + 0.001)
        
        # Relaxation steps
        success = False
        for _ in range(50): # Fixed number of relaxation steps per target
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = _compute_forces(points, r_target)
            
            # Update positions
            points = points + forces * step_size
            
            # Project to unit circle (hard constraint)
            dists = np.linalg.norm(points, axis=1)
            max_dist = 1.0 - r_target # Ideally inside
            # Actually, we just need to ensure they don't go out. 
            # The force handles the soft constraint, but we must clip hard.
            out_mask = dists > 1.0
            if np.any(out_mask):
                points[out_mask] = points[out_mask] / dists[out_mask, np.newaxis] * 0.999
        
        # Check if we achieved r_target
        current_r = _compute_radius(points)
        
        if current_r >= r_target - 1e-9:
            # Success
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
            # Keep step_size or reduce slightly
            step_size *= 0.99
        else:
            # Failure, revert
            points = best_points.copy()
            no_improve_count += 1
            # Reduce step size if stuck
            step_size *= 0.95
            if no_improve_count > 100:
                # Random restart or perturbation?
                # Just reduce target increase rate
                r_target = best_r * (1.0 + 0.0001)
                no_improve_count = 0
                
    return best_points

def _generate_hex_lattice(N, s=None):
    # Generate hex grid points centered at origin
    # Spacing s
    # We want to fit N points in radius 1.
    # Approximate s = 2 / sqrt(N) * 1.1 (safety factor)
    if s is None:
        s = 2.0 / np.sqrt(N) * 1.05
    
    max_k = int(1.0 / s) + 2
    points = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) >= N:
        # Sort by distance to origin, take closest N
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]
    else:
        # If not enough, shrink spacing and retry (should not happen with 1.05 factor)
        return _generate_hex_lattice(N, s * 0.9)

def _compute_radius(points):
    """
    Compute the maximum valid radius for the given points.
    r = min( min_pairwise_dist / 2, min_boundary_dist )
    """
    n = len(points)
    if n < 2:
        return 0.0
        
    # Pairwise distances
    # Use broadcasting for simplicity since N=300 is small
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # Boundary distance
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)

def _compute_forces(points, r_target):
    """
    Compute repulsive forces to resolve overlaps of circles with radius r_target.
    Force is proportional to overlap amount.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise forces
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    # Overlap = 2*r_target - dist
    # We only apply force if there is overlap
    overlap = 2.0 * r_target - dists
    overlap = np.where(overlap > 0, overlap, 0.0)
    
    # Normalize direction
    # force_dir[i,j] = (points[i] - points[j]) / dists[i,j]
    force_dirs = diff / dists_safe[:, :, np.newaxis]
    
    # Sum forces
    # F_i = sum_j overlap[i,j] * force_dirs[i,j]
    forces = np.sum(overlap[:, :, np.newaxis] * force_dirs, axis=1)
    
    # Boundary force
    # Push points inward if they are too close to boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_overlap = radii - (1.0 - r_target)
    boundary_overlap = np.where(boundary_overlap > 0, boundary_overlap, 0.0)
    
    # Direction is inward: -points / radii
    # Handle radii == 0
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    
    boundary_forces = -boundary_overlap[:, np.newaxis] * points / radii_safe[:, np.newaxis]
    
    forces += boundary_forces
    
    return forces

# EVOLVE-BLOCK-END
