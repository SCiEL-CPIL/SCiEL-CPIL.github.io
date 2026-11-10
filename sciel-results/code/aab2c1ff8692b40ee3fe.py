import numpy as np
import time

N = 100

def _log_energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    points: (N, 3) array of points on the unit sphere.
    Returns: (energy, gradient) where gradient is (N, 3).
    """
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We sum over all i != j and divide by 2
    # Mask for i != j
    mask = ~np.eye(N, dtype=bool)
    
    # log_dist[i, j] = log(dist[i, j])
    log_dist = np.log(dist)
    
    # Energy is -0.5 * sum_{i != j} log(dist[i, j])
    energy = -0.5 * np.sum(log_dist[mask])
    
    # Gradient calculation
    # grad_k = - sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    # Let's compute the term (x_k - x_j) / dist_sq[i, j]
    # We need to sum over j for each k.
    
    # inv_dist_sq = 1 / dist_sq
    inv_dist_sq = 1.0 / dist_sq
    
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[k] = - sum_j diff[k, j] * inv_dist_sq[k, j]
    # diff[k, j] is points[k] - points[j]
    # So grad[k] = - sum_j (points[k] - points[j]) * inv_dist_sq[k, j]
    
    # We can compute this as:
    # grad = - (diff * inv_dist_sq[..., np.newaxis]).sum(axis=1)
    
    grad = -(diff * inv_dist_sq[:, :, np.newaxis]).sum(axis=1)
    
    return energy, grad

def _project_to_sphere(points):
    """
    Project points onto the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial point: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = float('inf')
    best_points = points.copy()
    
    # Try a few initializations
    initializations = [points]
    
    # Add a random initialization
    rng = np.random.default_rng(42)
    random_points = rng.normal(size=(N, 3))
    random_points = _project_to_sphere(random_points)
    initializations.append(random_points)
    
    # Add a perturbed Fibonacci
    perturbed = points + rng.normal(scale=0.1, size=(N, 3))
    perturbed = _project_to_sphere(perturbed)
    initializations.append(perturbed)
    
    for init_points in initializations:
        if time.time() > deadline - 10:
            break
            
        current_points = init_points.copy()
        current_energy, _ = _log_energy_and_grad(current_points)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_points.copy()
            
        # Gradient Descent with Backtracking Line Search
        alpha = 0.1
        max_iter = 5000
        tol = 1e-10
        
        for iter in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            energy, grad = _log_energy_and_grad(current_points)
            
            if energy < best_energy:
                best_energy = energy
                best_points = current_points.copy()
                
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                break
                
            # Backtracking line search
            step_dir = -grad
            alpha_curr = alpha
            improved = False
            for _ in range(20):
                new_points = current_points + alpha_curr * step_dir
                new_points = _project_to_sphere(new_points)
                new_energy, _ = _log_energy_and_grad(new_points)
                
                if new_energy < energy:
                    current_points = new_points
                    improved = True
                    alpha = min(alpha * 1.5, 1.0) # Increase step size if successful
                    break
                else:
                    alpha_curr *= 0.5
                    
            if not improved:
                alpha = max(alpha * 0.5, 1e-6)
                if alpha < 1e-8:
                    break

    return best_points
