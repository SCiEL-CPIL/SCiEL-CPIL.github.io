"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a golden-angle spiral initialization followed by force-directed relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    # 1. Initialization: Golden Angle Spiral (Sunflower Model)
    # This distributes points evenly in the disk, respecting rotational symmetry.
    # r_i = sqrt(i/N), theta_i = i * golden_angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(N)
    # Avoid i=0 exactly at center to allow symmetric expansion, or keep it.
    # Standard sunflower:
    r = np.sqrt(indices / N)
    theta = indices * golden_angle
    
    # Scale r to stay well within boundary initially
    # Max r is 1.0. We want to start slightly inside to allow repulsion to push out.
    # Actually, starting at the boundary is fine if boundary force is attractive? 
    # No, boundary force is repulsive (keeps inside). 
    # If we start at r=1, force is infinite. 
    # Let's start at r * 0.9 to be safe, or use a soft boundary.
    # The force $1/(1-r)^2$ is strong.
    # Let's initialize at r * 0.95.
    r_init = r * 0.95
    x = r_init * np.cos(theta)
    y = r_init * np.sin(theta)
    points = np.column_stack((x, y))
    
    # 2. Force-Directed Relaxation
    # Objective: Maximize minimum distance.
    # Approximation: Maximize sum of inverse powers of distances.
    # Force F_ij = k * (c_i - c_j) / |c_i - c_j|^3 (repulsive)
    # Boundary Force F_b = k_b * c_i / |c_i| * 1 / (1 - |c_i|)^2 (repulsive from boundary)
    
    # Hyperparameters
    alpha = 0.01  # Initial step size
    alpha_decay = 0.999
    min_alpha = 1e-6
    force_power = 2.0  # Inverse square law
    boundary_power = 2.0
    
    # Precompute indices for vectorization
    # N=300, N^2 = 90,000. Vectorized operations are fast.
    
    best_score = -np.inf
    best_points = points.copy()
    
    # Compute score function
    def compute_score(pts):
        # min pairwise distance / 2
        # min distance to boundary
        # Vectorized pairwise distance
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Initial score
    current_score = compute_score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()

    iter_count = 0
    while time.time() < deadline:
        # Compute forces
        # Pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero for coincident points (shouldn't happen with spiral init)
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        
        # Force matrix F_ij: vector from j to i, scaled by 1/d^3
        # F_ij = diff / dist^3
        # We want repulsion: if i is close to j, push i away from j.
        # Vector diff is c_i - c_j.
        # Force on i due to j is proportional to (c_i - c_j) / |c_i - c_j|^3
        # Sum over j for each i.
        
        # Normalize diff by dist^3
        # dists_safe shape (N, N)
        # diff shape (N, N, 2)
        # factor shape (N, N, 1)
        factor = 1.0 / (dists_safe ** (force_power + 1))
        factor = np.where(np.eye(N, dtype=bool), 0.0, factor) # Zero out diagonal
        
        forces_pair = diff * factor[:, :, np.newaxis]
        net_force_pair = np.sum(forces_pair, axis=1)
        
        # Boundary force
        radii = np.linalg.norm(points, axis=1)
        # Avoid division by zero at center
        radii_safe = np.where(radii > 1e-10, radii, 1e-10)
        # Force direction: radial outward
        # Magnitude: 1 / (1 - r)^2
        # If r > 1, we are outside, force should push in? 
        # The problem says "inside a circle". 
        # If a point goes outside, we should pull it back or push it in.
        # The term (1 - r) becomes negative. (1 - r)^2 is positive.
        # Direction c/r is outward.
        # If r > 1, we want inward force.
        # Let's use a smooth barrier that works both ways or just project.
        # Simpler: if r >= 1, project back to r=0.99 and apply strong inward force.
        # For smooth optimization, let's use:
        # F_bound = - k * (points / radii) * (1 / (1 - radii)^2)
        # If r < 1, (1-r) is pos, force is inward (negative sign * outward dir).
        # Wait, repulsive from boundary means if I am close to boundary (r close to 1),
        # I should be pushed towards center.
        # Direction towards center is -points / radii.
        # So F_bound = k * (-points / radii) * (1 / (1 - radii)^2)
        
        # If r > 1, (1-r) is negative. Square is positive.
        # Force is still towards center. This is good.
        
        boundary_magnitude = 1.0 / ((1.0 - radii_safe) ** boundary_power)
        # Handle r=0 case: direction is undefined, force should be 0 or arbitrary.
        # With spiral init, no point is exactly at 0 unless N is small or we put one there.
        # Golden angle puts i=0 at r=0.
        # Let's handle r=0 by setting force to 0.
        boundary_dir = points / radii_safe[:, np.newaxis]
        boundary_dir = np.where(radii_safe[:, np.newaxis] < 1e-10, 0.0, boundary_dir)
        
        net_force_bound = -boundary_dir * boundary_magnitude[:, np.newaxis]
        
        total_force = net_force_pair + net_force_bound
        
        # Update positions
        points = points + alpha * total_force
        
        # Constrain points to stay inside the unit circle (hard constraint for stability)
        # If any point goes outside, pull it back to the surface (or slightly inside)
        new_radii = np.linalg.norm(points, axis=1)
        outside_mask = new_radii > 0.999
        if np.any(outside_mask):
            # Project back to 0.999
            scale = 0.999 / new_radii[outside_mask]
            points[outside_mask] *= scale[:, np.newaxis]
            
        # Compute score
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Decay step size
        alpha *= alpha_decay
        
        iter_count += 1
        
        # If alpha is too small, we might be stuck. 
        # We can restart with a different initialization if time permits.
        if alpha < min_alpha:
            # Try a different initialization?
            # For now, break to save time, or reset alpha for a few more iterations.
            # Let's break to return best found.
            break

    return best_points
# EVOLVE-BLOCK-END
