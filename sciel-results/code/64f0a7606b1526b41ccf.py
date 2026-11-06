"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    points: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad has the same shape as points.
    """
    n = len(points)
    # Compute pairwise distances
    # dist[i,j] = |points[i] - points[j]|
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0) on diagonal
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use only upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dx_i sum_{j != i} log(1/|x_i - x_j|)
    # = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # For each i: grad_i = sum_{j != i} (x_i - x_j) / dist(i,j)^2
    # Note: the energy is sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # d/dx_i [-log(d_ij)] = -(1/d_ij) * d(d_ij)/dx_i = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # So grad_i = -sum_{j != i, j > i} (x_i - x_j)/d_ij^2 - sum_{j != i, j < i} (x_i - x_j)/d_ij^2
    # Wait, let's be careful. Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} d(log d_ij)/dx_i - sum_{j<i} d(log d_ij)/dx_i
    # d(log d_ij)/dx_i = (1/d_ij) * (x_i - x_j)/d_ij = (x_i - x_j)/d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # But this is the gradient in R^3. We need to project to the tangent space of the sphere.
    # Actually, for the optimization on the sphere, we can use the Euclidean gradient
    # and then project, or use a Riemannian gradient.
    # Let's compute the Euclidean gradient first.
    
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d2 = dist_sq[i, j]
            grad[i] -= (points[i] - points[j]) / d2
    
    # This is O(N^2) which is fine for N=100
    return energy, grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Compute initial energy
    best_energy, _ = _log_energy_and_grad(points)
    best_points = points.copy()
    
    # Molecular dynamics with velocity Verlet
    # Forces: F_i = -grad_i (Euclidean)
    # But we need to constrain to the sphere. We'll use a simple approach:
    # integrate in R^3 and project back to sphere each step.
    
    velocity = np.zeros_like(points)
    dt = 0.001
    temperature = 0.1
    cooling_rate = 0.999
    
    # Adaptive step size
    max_dt = 0.01
    min_dt = 1e-5
    
    for step in range(100000):
        if time.time() > deadline:
            break
        
        # Compute forces (negative gradient)
        _, grad = _log_energy_and_grad(points)
        forces = -grad
        
        # Add thermal noise
        if temperature > 1e-8:
            forces += temperature * np.random.randn(*points.shape)
        
        # Velocity Verlet integration
        velocity += 0.5 * dt * forces
        points += dt * velocity
        points = _project_to_sphere(points)
        
        # Recompute forces at new position
        _, grad_new = _log_energy_and_grad(points)
        forces_new = -grad_new
        if temperature > 1e-8:
            forces_new += temperature * np.random.randn(*points.shape)
        
        velocity = 0.5 * (velocity + dt * forces_new)
        
        # Check energy
        energy, _ = _log_energy_and_grad(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Adaptive cooling
        temperature *= cooling_rate
        
        # Adaptive dt: reduce if energy increases too much
        if energy > best_energy + 0.1:
            dt = max(min_dt, dt * 0.9)
        else:
            dt = min(max_dt, dt * 1.01)
    
    # Final polish with L-BFGS-B on the sphere
    # We'll use a simple gradient descent with projection
    points = best_points.copy()
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline:
            break
        
        energy, grad = _log_energy_and_grad(points)
        
        # Project gradient to tangent space
        # tangent_grad_i = grad_i - (grad_i · x_i) x_i
        tangent_grad = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Gradient descent
        points = points - lr * tangent_grad
        points = _project_to_sphere(points)
        
        # Adaptive learning rate
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            lr = min(0.1, lr * 1.05)
        else:
            lr = max(1e-4, lr * 0.9)
    
    return best_points
# EVOLVE-BLOCK-END
