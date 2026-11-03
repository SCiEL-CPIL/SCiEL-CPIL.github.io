"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper function to compute the covering radius using a coarse grid
    def covering_radius_coarse(pts):
        n_grid = 40
        theta = np.linspace(0, np.pi, n_grid)
        phi_grid = np.linspace(0, 2*np.pi, n_grid)
        theta_grid, phi_grid = np.meshgrid(theta, phi_grid)
        test_points = np.column_stack([
            np.sin(theta_grid) * np.cos(phi_grid).flatten(),
            np.sin(theta_grid) * np.sin(phi_grid).flatten(),
            np.cos(theta_grid).flatten()
        ])
        diffs = test_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        max_angle = np.arccos(np.clip(1 - min_dists**2 / 2, -1, 1))
        return np.max(max_angle) * 180 / np.pi
    
    # Initial covering radius
    current_radius = covering_radius_coarse(points)
    
    # Simple iterative refinement
    n_iterations = 30
    step_size = 0.1
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
            
        new_points = points.copy()
        for i in range(N):
            if time.time() > deadline:
                break
            # Find 6 nearest neighbors
            dists = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.argsort(dists)[1:7]
            if len(neighbors) >= 6:
                centroid = np.mean(points[neighbors], axis=0)
                centroid /= np.linalg.norm(centroid)
                new_points[i] = points[i] + step_size * (centroid - points[i])
                new_points[i] /= np.linalg.norm(new_points[i])
        
        # Check if improved
        new_radius = covering_radius_coarse(new_points)
        if new_radius < current_radius:
            points = new_points
            current_radius = new_radius
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    return points
# EVOLVE-BLOCK-END
