"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _log_energy_grad(points):
    """Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = points[i] - points[j]
            dist2 = np.dot(d, d)
            if dist2 < 1e-30:
                continue
            grad[i] += d / dist2
    return grad

def _project_to_sphere(points):
    """Normalize points to unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _stochastic_lloyd(points, n_iter=200, temperature=0.1, rng=None):
    """Stochastic Lloyd's method for spherical Voronoi centroid approximation"""
    if rng is None:
        rng = np.random.default_rng(42)
    
    points = _project_to_sphere(points.copy())
    n = len(points)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for it in range(n_iter):
        # Decay temperature
        temp = temperature * (0.99 ** it)
        
        # For each point, compute its cell and move toward centroid
        new_points = points.copy()
        for i in range(n):
            # Find points in cell of i (closer to i than to any other)
            dists = np.array([np.linalg.norm(points[i] - points[j]) for j in range(n) if j != i])
            # A point j is in cell of i if dist(i,j) < dist(k,j) for all k != i,j
            # Simplified: use nearest neighbor approach
            # Actually, for Lloyd's, we want the centroid of the Voronoi cell
            # Approximate: move toward average of nearest neighbors weighted by inverse distance
            
            # Compute distances from point i to all others
            dists_from_i = np.array([np.linalg.norm(points[i] - points[j]) for j in range(n) if j != i])
            indices = [j for j in range(n) if j != i]
            
            # Weight by inverse squared distance (like RBF)
            weights = 1.0 / (dists_from_i ** 2 + 1e-10)
            weights /= np.sum(weights)
            
            # Centroid of weighted neighbors
            neighbors = points[indices]
            centroid = np.sum(neighbors * weights[:, np.newaxis], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-15)
            
            # Stochastic update
            noise = rng.normal(0, temp, size=3)
            new_points[i] = points[i] + 0.5 * (centroid - points[i]) + noise
        
        new_points = _project_to_sphere(new_points)
        
        # Check if improved
        energy = _log_energy(new_points)
        if energy < best_energy:
            best_energy = energy
            best_points = new_points.copy()
        
        points = new_points
    
    return best_points, best_energy

def _coordinate_repulsion(points, n_iter=500, tol=1e-12):
    """Gauss-Seidel coordinate repulsion refinement"""
    points = _project_to_sphere(points.copy())
    n = len(points)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for it in range(n_iter):
        improved = False
        for i in range(n):
            for coord in range(3):
                # Try small perturbations in this coordinate
                best_val = points[i, coord]
                best_local_energy = _log_energy(points)
                
                # Use gradient-based step
                grad = _log_energy_grad(points)
                # Project gradient to tangent space
                grad_tangent = grad[i] - np.dot(grad[i], points[i]) * points[i]
                
                # Try moving along negative gradient
                step_size = 0.01
                for attempt in range(10):
                    new_point = points[i].copy()
                    new_point[coord] += step_size * grad_tangent[coord]
                    new_point = new_point / (np.linalg.norm(new_point) + 1e-15)
                    
                    points[i] = new_point
                    local_energy = _log_energy(points)
                    
                    if local_energy < best_local_energy:
                        best_local_energy = local_energy
                        best_val = new_point[coord]
                        improved = True
                    else:
                        break
                    step_size *= 0.5
                
                points[i, coord] = best_val
        
        energy = _log_energy(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        if not improved and it > 50:
            break
    
    return best_points, best_energy

def _gradient_descent_polish(points, n_iter=1000, lr_init=0.01):
    """Final gradient-based polish with adaptive learning rate"""
    points = _project_to_sphere(points.copy())
    n = len(points)
    best_points = points.copy()
    best_energy = _log_energy(points)
    lr = lr_init
    
    for it in range(n_iter):
        grad = _log_energy_grad(points)
        
        # Project gradient to tangent space for each point
        grad_tangent = np.array([
            grad[i] - np.dot(grad[i], points[i]) * points[i] 
            for i in range(n)
        ])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-15:
            break
        
        # Update
        points_new = points - lr * grad_tangent
        points_new = _project_to_sphere(points_new)
        
        energy = _log_energy(points_new)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points_new.copy()
            points = points_new
            # Decay learning rate
            lr *= 0.99
        else:
            # Reduce learning rate on failure
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_points, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy"""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    # Phase 1: Stochastic Lloyd's for global structure
    if time.time() < deadline - 30:
        lloyd_points, lloyd_energy = _stochastic_lloyd(points, n_iter=100, temperature=0.05, rng=np.random.default_rng(123))
        if lloyd_energy < best_energy:
            best_energy = lloyd_energy
            best_points = lloyd_points
    
    # Phase 2: Coordinate repulsion for local refinement
    if time.time() < deadline - 20:
        coord_points, coord_energy = _coordinate_repulsion(best_points, n_iter=300)
        if coord_energy < best_energy:
            best_energy = coord_energy
            best_points = coord_points
    
    # Phase 3: Gradient polish for final precision
    if time.time() < deadline - 5:
        polish_points, polish_energy = _gradient_descent_polish(best_points, n_iter=500, lr_init=0.005)
        if polish_energy < best_energy:
            best_energy = polish_energy
            best_points = polish_points
    
    # Final projection to ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-9:
                # Add small perturbation
                best_points[j] += np.random.default_rng(456).normal(0, 1e-6, size=3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

