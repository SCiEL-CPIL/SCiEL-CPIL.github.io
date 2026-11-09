"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 200 equal non-overlapping circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between centers and from boundaries.
    
    Strategy:
    1. Initialize centers on a hexagonal lattice pattern.
    2. Use force-based relaxation (repulsive forces) to push circles apart
       and toward local optimality.
    """
    # Hexagonal lattice initialization
    # Hexagonal packing has density pi/(2*sqrt(3)) ~ 0.9069
    # For N circles, we want a roughly hexagonal arrangement
    
    # Estimate the number of rows and columns for hexagonal packing
    # In hexagonal packing, rows alternate with offset
    # Let's try to create a hexagonal grid that fits ~200 points
    
    # Hexagonal lattice parameters
    # Row spacing: dy
    # Column spacing: dx
    # For hexagonal: dx = r_hex * 2, dy = r_hex * sqrt(3)
    # where r_hex is the "radius" of the hexagonal cell
    
    # We want to fit 200 points in a 1x1 square
    # Let's try different aspect ratios
    
    # Start with a rough hexagonal arrangement
    # Number of rows
    n_rows = int(np.sqrt(N * np.sqrt(3) / 2))
    if n_rows < 1:
        n_rows = 1
    
    # Points per row
    points_per_row = int(np.ceil(N / n_rows))
    
    # Create hexagonal lattice
    centers = []
    for i in range(n_rows):
        row_points = points_per_row
        # Offset every other row
        offset = 0.5 if (i % 2 == 1) else 0.0
        for j in range(row_points):
            x = (j + offset) / (points_per_row - 0.5) - 0.5
            y = (i + 0.5) / n_rows - 0.5
            # Clamp to square
            x = np.clip(x, -0.48, 0.48)
            y = np.clip(y, -0.48, 0.48)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # Force-based relaxation
    # Repulsive force between pairs: F = k / d^2 (capped)
    # Attractive force to keep them in the square (boundary repulsion)
    
    def objective_and_forces(pos):
        pos = pos.reshape(-1, 2)
        # Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Min distance
        min_dist = np.min(dist)
        
        # Boundary distances
        boundary_dists = []
        for k in range(2):
            boundary_dists.append(0.5 - np.abs(pos[:, k]))
        min_boundary = np.min(boundary_dists)
        
        # Score is min of these
        score = min(min_dist / 2, min_boundary)
        
        # Forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsive forces
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < 1e-10:
                    # Random push
                    forces[i, 0] += 0.01 * (np.random.rand() - 0.5)
                    forces[i, 1] += 0.01 * (np.random.rand() - 0.5)
                    forces[j, 0] -= 0.01 * (np.random.rand() - 0.5)
                    forces[j, 1] -= 0.01 * (np.random.rand() - 0.5)
                    continue
                # Force magnitude: stronger when closer
                f_mag = 1.0 / (d * d + 1e-6)
                # Cap the force
                f_mag = min(f_mag, 10.0)
                # Direction: from j to i for particle i
                direction = (pos[i] - pos[j]) / d
                forces[i] += f_mag * direction
                forces[j] -= f_mag * direction
        
        # Boundary forces: push away from boundaries
        for i in range(N):
            for k in range(2):
                if pos[i, k] < -0.48:
                    forces[i, k] += 5.0 * (-0.48 - pos[i, k])
                elif pos[i, k] > 0.48:
                    forces[i, k] -= 5.0 * (pos[i, k] - 0.48)
        
        return score, forces
    
    # Iterative relaxation
    pos = centers.copy()
    step_size = 0.01
    n_iterations = 500
    
    for it in range(n_iterations):
        score, forces = objective_and_forces(pos)
        
        # Update positions
        pos += step_size * forces
        
        # Clamp to square with small margin
        pos = np.clip(pos, -0.49, 0.49)
        
        # Decrease step size over time
        step_size *= 0.995
    
    # Final optimization using scipy
    def negative_score(pos):
        pos = pos.reshape(-1, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        boundary_dists = []
        for k in range(2):
            boundary_dists.append(0.5 - np.abs(pos[:, k]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_dist / 2, min_boundary)
        return -score
    
    # Use Nelder-Mead for local optimization
    pos_flat = pos.flatten()
    
    # Run a few rounds of optimization
    for _ in range(3):
        result = minimize(
            negative_score,
            pos_flat,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        pos_flat = result.x
        pos = pos_flat.reshape(-1, 2)
        # Clamp
        pos = np.clip(pos, -0.49, 0.49)
    
    return pos
# EVOLVE-BLOCK-END
