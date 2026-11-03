import numpy as np
import math
import time

N = 150

def compute_energy(points):
    """
    Compute the logarithmic energy sum_{i<j} log(1 / |x_i - x_j|).
    Returns energy value.
    """
    # points shape (N, 3)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dists = np.sqrt(dist_sq)
    
    # We need sum over i < j
    # Mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    # Energy is sum(log(1/d)) = -sum(log(d))
    energy = -np.sum(np.log(pair_dists))
    return energy

def compute_forces(points):
    """
    Compute the negative gradient of the energy (forces) for each point.
    F_i = - dE/dx_i
    E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j != i} d/dx_i log(|x_i - x_j|)
    d/dx_i log(|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    So dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Force F_i = - dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    dist_sq[dist_sq < 1e-30] = 1e-30
    
    # inv_dist_sq[i, j] = 1 / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    # Set diagonal to 0 to exclude self-interaction
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # forces[i, j] = (x_i - x_j) * inv_dist_sq[i, j]
    # Force on i from j is (x_i - x_j) / |x_i - x_j|^2
    # Total force on i is sum over j of (x_i - x_j) / |x_i - x_j|^2
    forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return forces

def project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-15] = 1.0
    return points / norms

def local_refine(points, steps=100, lr=0.01):
    """
    Perform a few steps of Gradient Descent on the sphere to polish the solution.
    """
    for _ in range(steps):
        forces = compute_forces(points)
        # Move against the gradient (which is the force direction for minimization? 
        # Wait. Force F = -grad E. So moving in direction of F decreases E.
        # x_new = x + lr * F
        points = points + lr * forces
        points = project_to_sphere(points)
    return points

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # 1. Initialization: Fibonacci Sphere for good spread
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = compute_energy(best_points)
    
    # 2. Langevin Dynamics / Simulated Annealing
    # Parameters
    alpha = 0.01 # Step size
    beta = 0.1   # Noise level
    decay = 0.995
    
    rng = np.random.default_rng(42)
    
    # We will run in loops until time limit
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Compute forces
        forces = compute_forces(points)
        
        # Generate noise
        noise = rng.normal(size=(N, 3))
        
        # Update
        points = points + alpha * forces + beta * noise
        points = project_to_sphere(points)
        
        # Calculate energy
        current_energy = compute_energy(points)
        
        # Acceptance criterion (Metropolis)
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
            
        # Cooling
        alpha *= decay
        beta *= decay
        
        # If alpha becomes too small, we might stagnate. 
        # We can restart or just stop if improvement is negligible.
        if alpha < 1e-6:
            break

    # 3. Final Local Refinement
    # Use the best points found so far
    refined_points = local_refine(best_points, steps=200, lr=0.005)
    refined_energy = compute_energy(refined_points)
    
    if refined_energy < best_energy:
        best_points = refined_points
        best_energy = refined_energy
        
    # One more round of refinement with smaller step
    refined_points2 = local_refine(best_points, steps=100, lr=0.001)
    refined_energy2 = compute_energy(refined_points2)
    
    if refined_energy2 < best_energy:
        best_points = refined_points2
        best_energy = refined_energy2
        
    return best_points
