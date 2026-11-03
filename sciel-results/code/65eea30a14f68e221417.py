"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

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

def _compute_forces(points):
    """
    Compute repulsive forces for logarithmic energy minimization.
    Force on i from j is (x_i - x_j) / ||x_i - x_j||^2.
    """
    n = len(points)
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    
    # factor[i,j] = 1 / d2[i,j]
    factor = 1.0 / d2
    np.fill_diagonal(factor, 0.0)
    
    # force[i] = sum_j (points[i] - points[j]) / d2[i,j]
    force = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot_prod = np.sum(force * points, axis=1)
    tangent_force = force - dot_prod[:, np.newaxis] * points
    
    return tangent_force

def _sa_step(points, temp, alpha):
    """
    Perform one step of Simulated Annealing / Force-Directed Layout.
    """
    forces = _compute_forces(points)
    
    # Move points
    new_points = points + alpha * forces
    
    # Add noise to escape local minima
    if temp > 1e-6:
        noise = np.random.normal(0, temp, size=points.shape)
        new_points = new_points + noise
        
    # Project to sphere
    new_points = _project_to_sphere(new_points)
    
    return new_points

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_energy = float('inf')
    
    # Initial guess
    init_points = _fibonacci_sphere(N)
    
    # Multiple restarts with different initial conditions and SA parameters
    np.random.seed(42)
    
    num_restarts = 5
    max_iters = 10000
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            current = init_points.copy()
        else:
            # Random perturbation
            noise_std = 0.5
            current = init_points + np.random.normal(0, noise_std, size=(N, 3))
            current = _project_to_sphere(current)
            
        current_energy = _log_energy(current)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current.copy()
            
        # Simulated Annealing
        temp_start = 0.1
        temp_end = 1e-6
        cooling_rate = (temp_end / temp_start) ** (1.0 / max_iters)
        temp = temp_start
        alpha = 0.1
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            current = _sa_step(current, temp, alpha)
            temp *= cooling_rate
            
            # Occasionally re-evaluate energy to check for improvements
            # Doing it every 100 steps to save time
            if it % 100 == 0 or it == max_iters - 1:
                current_energy = _log_energy(current)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current.copy()
                    
        # Refinement with Gradient Descent
        # Use the best point from SA as starting point for GD
        current = best_points.copy()
        current_energy = best_energy
        
        max_gd_iters = 500
        alpha_gd = 0.1
        c1 = 1e-4
        rho = 0.5
        
        for it in range(max_gd_iters):
            if time.time() - start_time > time_limit:
                break
                
            grad = _compute_forces(current) # Reusing force function as negative gradient
            # Note: _compute_forces returns the negative gradient (force to minimize energy)
            # So we move in the direction of the force.
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
                
            step = alpha_gd * grad
            new_points = _project_to_sphere(current + step)
            new_energy = _log_energy(new_points)
            
            expected_decrease = c1 * alpha_gd * grad_norm * grad_norm
            
            while new_energy > current_energy - expected_decrease and alpha_gd > 1e-8:
                alpha_gd *= rho
                step = alpha_gd * grad
                new_points = _project_to_sphere(current + step)
                new_energy = _log_energy(new_points)
                expected_decrease = c1 * alpha_gd * grad_norm * grad_norm
                
            if alpha_gd <= 1e-8:
                break
                
            current = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current.copy()
                
            # Reset alpha if step was full
            if new_energy <= current_energy - c1 * alpha_gd * grad_norm * grad_norm:
                alpha_gd = min(alpha_gd * 1.2, 0.5)
            else:
                alpha_gd = alpha_gd * 0.8

    return best_points
# EVOLVE-BLOCK-END
