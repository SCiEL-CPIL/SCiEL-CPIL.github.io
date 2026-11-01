import numpy as np
import time

N = 100

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its Euclidean gradient.
    Energy: sum_{i<j} log(1/|xi - xj|) = - sum_{i<j} log|xi - xj|
    Grad_i: sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    # points shape (N, 3)
    # Compute pairwise distances
    # diff[i, j] = xi - xj
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dist2[i, j] = |xi - xj|^2
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: -0.5 * sum_{i!=j} log(sqrt(dist2)) = -0.25 * sum_{i!=j} log(dist2)
    # Or sum_{i<j} -log(sqrt(dist2)) = -0.5 sum_{i<j} log(dist2)
    # Let's use the symmetric sum for easier gradient calculation.
    # E = -0.5 * sum_{i!=j} log(0.5 * dist2) ? No.
    # log(1/d) = -log(d).
    # E = sum_{i<j} -log(d_ij)
    
    # Gradient of E w.r.t xi:
    # d/dxi (-log|xi-xj|) = -(1/|xi-xj|) * (xi-xj)/|xi-xj| = -(xi-xj)/|xi-xj|^2
    # Wait, the energy is log(1/d) = -log(d).
    # Derivative of -log(d) wrt xi is -1/d * d/dxi d = -1/d * (xi-xj)/d = -(xi-xj)/d^2.
    # So the force is attractive? No, we want to MINIMIZE energy.
    # If points are close, d is small, -log(d) is large positive.
    # We want to maximize d.
    # So we want to move points APART.
    # The gradient of the energy points in the direction of INCREASING energy.
    # We want to DECREASE energy, so we move AGAINST the gradient.
    # Grad E_i = sum_{j!=i} -(xi-xj)/|xi-xj|^2 = sum_{j!=i} (xj-xi)/|xi-xj|^2.
    # This vector points from xi towards xj.
    # Moving xi in direction -Grad E_i means moving xi away from xj. Correct.
    
    # Let's compute the gradient matrix.
    # grad[i, j] = (xi - xj) / dist2[i, j]
    # We want sum_{j!=i} (xi - xj) / dist2[i, j]
    # Note: (xi - xj) / dist2 is the vector.
    
    # diff is xi - xj.
    # We need to divide each vector diff[i,j] by dist2[i,j].
    # dist2 shape (N, N). diff shape (N, N, 3).
    # We can do diff / dist2[:,:,np.newaxis]
    
    # Set diagonal to 0 to exclude self-interaction
    mask = ~np.eye(N, dtype=bool)
    diff_masked = np.where(mask[:,:,np.newaxis], diff, 0.0)
    dist2_safe = np.where(mask, dist2, 1.0)
    
    grad_vec = diff_masked / dist2_safe[:,:,np.newaxis]
    
    # Sum over j (axis 1) to get gradient for each i
    grad = np.sum(grad_vec, axis=1)
    
    # Energy
    # E = sum_{i<j} -log(sqrt(dist2[i,j])) = -0.5 sum_{i<j} log(dist2[i,j])
    # sum_{i<j} log(dist2) = 0.5 * (sum_{i!=j} log(dist2))
    # So E = -0.25 * sum_{i!=j} log(dist2[i,j])
    # Let's calculate sum_{i!=j} log(dist2)
    log_dist2 = np.log(dist2_safe)
    # Sum over all i!=j
    total_log_dist2 = np.sum(log_dist2[mask])
    energy = -0.25 * total_log_dist2
    
    return energy, grad

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    energy, grad = _energy_and_grad(points)
    best_energy = energy
    best_points = points.copy()
    
    step_size = 1e-3
    min_step_size = 1e-8
    
    while time.time() - start_time < deadline:
        # Project gradient to tangent space
        # grad_t = grad - (grad . points) * points
        dot_prod = np.sum(grad * points, axis=1)
        grad_t = grad - dot_prod[:, np.newaxis] * points
        
        # Update
        new_points = points - step_size * grad_t
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, np.newaxis]
        
        new_energy, new_grad = _energy_and_grad(new_points)
        
        if new_energy < energy:
            points = new_points
            energy = new_energy
            grad = new_grad
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
            # Increase step size slightly if stable?
            step_size = min(step_size * 1.05, 1e-2)
        else:
            step_size *= 0.5
            if step_size < min_step_size:
                break
        
    return best_points
