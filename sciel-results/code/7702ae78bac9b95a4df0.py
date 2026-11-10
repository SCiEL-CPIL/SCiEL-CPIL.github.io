"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Place 200 equal circles in a unit square [-0.5, 0.5]^2 to maximize the
    minimum radius. Uses a Fibonacci lattice initialization followed by a
    repulsive force-based simulation (Langevin dynamics with rigid constraints).
    """
    np.random.seed(42)
    
    # Step 1: Fibonacci lattice (sunflower pattern) initialization
    # This gives a well-distributed starting point
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 rad
    # Map Fibonacci points to the square
    # Standard sunflower: r = sqrt(i/n), theta = i * golden_angle
    # Scale to fit in [-0.5, 0.5]^2
    i = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(i / n) * 0.5  # scale so max r is 0.5
    theta = i * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Clip to square to ensure validity
    x = np.clip(x, -0.5, 0.5)
    y = np.clip(y, -0.5, 0.5)
    
    centres = np.column_stack([x, y])
    
    # Step 2: Repulsive force-based simulation
    # We want to maximize the minimum distance between any two centres,
    # and also from the boundaries.
    
    # Parameters for the simulation
    n_iter = 5000
    dt = 0.001
    k_repulse = 100.0  # repulsion strength between circles
    k_boundary = 50.0  # repulsion from boundaries
    damping = 0.95  # velocity damping
    
    velocities = np.zeros_like(centres)
    
    # Precompute for efficiency
    for iteration in range(n_iter):
        forces = np.zeros_like(centres)
        
        # Compute pairwise repulsive forces
        # For each pair (i, j), if they are too close, push them apart
        # Use a simple O(N^2) approach since N=200 is small
        for i in range(n):
            for j in range(i + 1, n):
                diff = centres[i] - centres[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.array([np.random.randn(), np.random.randn()]) * 0.01
                    forces[j] -= np.array([np.random.randn(), np.random.randn()]) * 0.01
                    continue
                # Repulsive force: stronger when closer
                # Use a smooth repulsion that dominates at small distances
                force_mag = k_repulse / (dist * dist + 1e-6)
                force_dir = diff / dist
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        # Boundary forces: push centres away from edges
        # The constraint is that centres must be in [-0.5, 0.5]^2
        # We want to maximize the distance from the boundary
        for i in range(n):
            for dim in range(2):
                # Left boundary
                if centres[i, dim] < -0.45:
                    forces[i, dim] += k_boundary * (-0.45 - centres[i, dim])
                # Right boundary
                if centres[i, dim] > 0.45:
                    forces[i, dim] -= k_boundary * (centres[i, dim] - 0.45)
                # Top boundary
            # Actually, let's be more careful with boundary handling
            # We want to keep centres well inside, so use a softer boundary
            # The actual constraint is |c_ik| <= 0.5 - r, but we don't know r yet
            # So we just keep them in [-0.48, 0.48] as a safety margin
        
        # Update velocities and positions
        velocities = damping * velocities + dt * forces
        centres = centres + dt * velocities
        
        # Hard constraint: keep centres within the square
        centres = np.clip(centres, -0.49, 0.49)
    
    # Final step: compute the actual radius and verify validity
    # The radius is min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    
    # Compute minimum pairwise distance
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Compute minimum distance to boundary
    min_boundary_dist = np.inf
    for i in range(n):
        for k in range(2):
            d = 0.5 - abs(centres[i, k])
            if d < min_boundary_dist:
                min_boundary_dist = d
    
    r = min(min_pair_dist / 2, min_boundary_dist)
    
    return centres
# EVOLVE-BLOCK-END
