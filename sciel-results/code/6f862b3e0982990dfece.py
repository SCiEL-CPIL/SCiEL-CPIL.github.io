"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    best_points = None
    best_energy = float('inf')
    
    # Initial guess: Fibonacci spiral
    init_points = _fibonacci_sphere(N)
    
    # We will run several independent optimization runs
    # Each run uses a slightly perturbed starting point and an adaptive step size.
    
    num_restarts = 5
    num_iterations = 2000
    
    # Fixed seed for reproducibility in restarts
    np.random.seed(123)
    
    for restart in range(num_restarts):
        if restart == 0:
            current = init_points.copy()
        else:
            # Add random noise and project back
            noise_std = 0.1
            current = init_points + np.random.normal(0, noise_std, size=(N, 3))
            current = _project_to_sphere(current)
            
        for it in range(num_iterations):
            # Decay learning rate
            progress = it / num_iterations
            lr = 0.1 * (1.0 - progress) ** 2 + 1e-6
            
            # Vectorized force calculation
            # diff[i,j] = current[i] - current[j]
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            
            # d2 = sum(diff^2, axis=2)
            d2 = np.sum(diff * diff, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(d2, 1.0)
            
            # Force factor: 1 / d2
            factor = 1.0 / d2
            
            # Zero out diagonal
            np.fill_diagonal(factor, 0.0)
            
            # force[i] = sum_j (diff[i,j] * factor[i,j])
            force = np.sum(diff * factor[:, :, np.newaxis], axis=1)
            
            # Project force to tangent plane
            dot_prod = np.sum(force * current, axis=1)
            tangent_force = force - dot_prod[:, np.newaxis] * current
            
            # Update
            current = current + lr * tangent_force
            
            # Project to sphere
            norms = np.linalg.norm(current, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            current = current / norms
            
        energy = _log_energy(current)
        if energy < best_energy:
            best_energy = energy
            best_points = current.copy()
            
    return best_points
# EVOLVE-BLOCK-END
