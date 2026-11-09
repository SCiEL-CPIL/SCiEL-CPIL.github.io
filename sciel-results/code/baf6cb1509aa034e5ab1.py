"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Initialize sphere centers using a deterministic structured lattice
    # to ensure maximal initial separation, then refine positions via
    # a repulsion-based relaxation algorithm and a final Powell optimization
    # to maximize the minimum pairwise distance and boundary clearance.
    
    def grid_initialization(n):
        """Generate n points in a unit cube using a grid-based approach."""
        # Use a 4x4x4 grid (64 points) and select n points
        # This provides a good initial separation (approx 1/3)
        grid_size = 4
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = (i + 0.5) / grid_size - 0.5
                    y = (j + 0.5) / grid_size - 0.5
                    z = (k + 0.5) / grid_size - 0.5
                    points.append([x, y, z])
        
        points = np.array(points)
        
        # We need exactly n points. 
        # Use a fixed seed for deterministic selection
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(points))
        points = points[indices[:n]]
        
        return points

    def relaxation(centers, iterations=5000):
        """Refine positions using a repulsion-based relaxation."""
        centers = centers.copy()
        # Step size decay
        step_size = 0.01
        min_step = 1e-5
        
        for it in range(iterations):
            # Decay step size
            step_size = max(min_step, step_size * 0.999)
            
            # Calculate forces for each point
            for i in range(N):
                # Calculate distance to all other points
                diffs = centers[i] - centers
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf # Exclude self
                
                # Find nearest neighbor
                min_dist_idx = np.argmin(dists)
                min_dist = dists[min_dist_idx]
                
                # Direction away from nearest neighbor
                if min_dist > 1e-9:
                    force = (centers[i] - centers[min_dist_idx]) / min_dist
                else:
                    force = np.zeros(3)
                
                # Boundary repulsion
                for axis in range(3):
                    coord = centers[i, axis]
                    dist_to_face = 0.5 - abs(coord)
                    if dist_to_face < min_dist:
                        # Push away from boundary (towards center)
                        normal = -np.sign(coord)
                        if coord == 0:
                            normal = 1.0
                        force[axis] += normal * (1.0 - dist_to_face)
                        
                # Update position
                norm_force = np.linalg.norm(force)
                if norm_force > 1e-9:
                    centers[i] += force / norm_force * step_size
                
                # Clip to cube
                centers[i] = np.clip(centers[i], -0.5, 0.5)
                
        return centers

    def objective(params):
        """Objective to minimize: negative of the minimum clearance."""
        centers = params.reshape(-1, 3)
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Compute boundary clearance
        boundary_clearance = np.min(0.5 - np.abs(centers))
        
        r = min(min_pair_dist / 2.0, boundary_clearance)
        
        # We want to maximize r, so minimize -r
        return -r
    
    # Initialize with grid
    centers_init = grid_initialization(N)
    
    # Relax
    centers_relaxed = relaxation(centers_init, iterations=5000)
    
    # Final polish with Powell optimization
    params_init = centers_relaxed.flatten()
    result = minimize(objective, params_init, method='Powell',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    # Extract optimized centers
    centers = result.x.reshape(-1, 3)
    
    # Ensure centers are within the cube (clip to valid range)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
