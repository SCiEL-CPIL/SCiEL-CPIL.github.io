import numpy as np
import time

N = 100

def construct():
    # Start time to enforce deadline
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to calculate energy
    def energy(pts):
        # Pairwise distances
        # Avoid division by zero
        d2 = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=-1)
        np.fill_diagonal(d2, 1.0) # Avoid log(0)
        d = np.sqrt(d2)
        # Sum log(1/d) = - sum log(d)
        # Only sum i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        return -np.sum(np.log(d[mask]))
    
    current_energy = energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Gradient Descent
    # Gradient of Energy E = - sum log(dist)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # We want to minimize E, so we move in direction -dE/dx_i
    # -dE/dx_i = sum_{j!=i} (x_j - x_i) / ||x_i - x_j||^2
    
    eta = 0.01  # Initial learning rate
    max_iter = 5000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(d2, 1.0)
        
        # Compute gradient
        # term_ij = (x_i - x_j) / d2_ij
        # We need sum over j for each i
        # grad_i = sum_j (x_i - x_j) / d2_ij
        # Note: diff[i,j] = x_i - x_j
        # So grad = sum_j diff[i,j] / d2[i,j]
        
        # Avoid division by zero (diagonal is 1.0, so no issue)
        grad = np.sum(diff / d2[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        # grad_tan = grad - (grad . x_i) x_i
        dot = np.sum(grad * points, axis=1)
        grad_tan = grad - dot[:, np.newaxis] * points
        
        # Update points
        new_points = points + eta * grad_tan
        
        # Normalize to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Calculate new energy
        new_energy = energy(new_points)
        
        # Check for improvement
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            # Optional: increase eta slightly if progress is good
            # eta = min(eta * 1.01, 0.1)
        else:
            # Decrease eta if energy increased
            eta *= 0.8
            if eta < 1e-6:
                break
                
    return best_points
