"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 150

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 150 points on the unit sphere minimizing logarithmic energy.
    Uses a multi-stage optimization:
    1. Start with a well-distributed initial configuration (Fibonacci spiral).
    2. Apply gradient-based local optimization using the repulsive force model.
    3. Use multiple random restarts to avoid local minima.
    """
    best_energy = np.inf
    best_points = None
    
    # Precompute for efficiency
    def compute_energy(points):
        """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
        # points: (N, 3) array of unit vectors
        # Compute all pairwise distances
        # Use broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        # Avoid log(0) by masking diagonal and very small distances
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        # Filter out any zero or near-zero distances
        dists = dists[dists > 1e-15]
        if len(dists) == 0:
            return np.inf
        # Energy is sum of log(1/d) = -sum of log(d)
        energy = -math.fsum(np.log(dists).tolist())
        return energy
    
    def compute_forces(points):
        """
        Compute the repulsive force on each point.
        Force on point i from point j: -grad_{xi} log(1/|xi-xj|) = -1/|xi-xj| * (xi-xj)/|xi-xj|^2 ... 
        Actually, d/dxi log(1/|xi-xj|) = d/dxi (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
        So the force (negative gradient of energy) is sum_j (xi-xj)/|xi-xj|^2 for j != i
        But we need to project onto the sphere tangent space.
        """
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-15:
                    continue
                dist = np.sqrt(dist_sq)
                # Force contribution: (xi - xj) / |xi - xj|^2
                f = diff / dist_sq
                forces[i] += f
        # Project forces onto tangent space of sphere
        for i in range(N):
            # Tangent projection: f - (f . xi) * xi
            dot = np.dot(forces[i], points[i])
            forces[i] -= dot * points[i]
        return forces
    
    def optimize_from_start(start_points, max_iter=5000, lr_init=0.01):
        """
        Local optimization using gradient descent with adaptive learning rate.
        """
        points = start_points.copy()
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        energy = compute_energy(points)
        
        # Adaptive learning rate
        lr = lr_init
        best_local_energy = energy
        best_local_points = points.copy()
        
        for it in range(max_iter):
            forces = compute_forces(points)
            
            # Gradient descent step (moving in direction of force = negative gradient)
            new_points = points + lr * forces
            
            # Project back to sphere
            new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / new_norms
            
            new_energy = compute_energy(new_points)
            
            if not np.isfinite(new_energy):
                lr *= 0.5
                continue
            
            if new_energy < energy:
                points = new_points
                energy = new_energy
                if energy < best_local_energy:
                    best_local_energy = energy
                    best_local_points = points.copy()
                # Increase learning rate slightly if making progress
                lr = min(lr * 1.05, 0.1)
            else:
                # Decrease learning rate
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        return best_local_points, best_local_energy
    
    # Generate multiple random starting configurations
    np.random.seed(42)
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Start 2-5: Random points with some structure
    starts = [fib_points]
    for restart in range(4):
        # Random points on sphere
        random_points = np.random.randn(N, 3)
        random_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)
        starts.append(random_points)
    
    # Optimize from each start
    for start in starts:
        opt_points, opt_energy = optimize_from_start(start, max_iter=3000, lr_init=0.005)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
    
    # Final polish with smaller learning rate
    if best_points is not None:
        final_points, final_energy = optimize_from_start(best_points, max_iter=5000, lr_init=0.001)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    # Ensure all points are on unit sphere and finite
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    # Validate
    if not np.all(np.isfinite(best_points)):
        # Fallback to Fibonacci spiral
        return fib_points
    
    return best_points
# EVOLVE-BLOCK-END
