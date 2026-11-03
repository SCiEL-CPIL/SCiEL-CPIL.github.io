import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _fibonacci_sphere(n, rng=None):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    if rng is not None:
        # Random rotation
        angles = rng.uniform(0, 2 * np.pi, 3)
        # Apply rotation matrices for x, y, z axes
        for axis, angle in enumerate(angles):
            if axis == 0:  # rotate around x
                c, s = np.cos(angle), np.sin(angle)
                y, z_ = X[:, 1].copy(), X[:, 2].copy()
                X[:, 1] = c * y - s * z_
                X[:, 2] = s * y + c * z_
            elif axis == 1:  # rotate around y
                c, s = np.cos(angle), np.sin(angle)
                x, z_ = X[:, 0].copy(), X[:, 2].copy()
                X[:, 0] = c * x + s * z_
                X[:, 2] = -s * x + c * z_
            else:  # rotate around z
                c, s = np.cos(angle), np.sin(angle)
                x, y = X[:, 0].copy(), X[:, 1].copy()
                X[:, 0] = c * x - s * y
                X[:, 1] = s * x + c * y
    return X

def _energy_and_grad(x):
    """
    Compute energy and gradient for logarithmic energy.
    x: flattened array of shape (3N,)
    Returns: energy, gradient
    """
    X = x.reshape(N, 3)
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy = sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # But we only want i<j, so:
    # E = sum_{i<j} -log(d_ij) = -0.5 * sum_{i!=j} -log(d_ij) = 0.5 * sum_{i!=j} log(d_ij)
    # Wait, let's be careful.
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
    # So E = -0.5 * sum_{i!=j} log(d_ij)
    
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # So E = -0.5 * sum_{i!=j} 0.5 * log(dist_sq_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    
    # Gradient: dE/dx_k = ?
    # E = -0.5 * sum_{i<j} log(d_ij)
    # dE/dx_k = -0.5 * sum_{i<j} (1/d_ij) * d(d_ij)/dx_k
    # d(d_ij)/dx_k = (x_k - x_j)/d_ij if k=i, (x_k - x_i)/d_ij if k=j
    # So dE/dx_k = -0.5 * sum_{j!=k} (x_k - x_j)/d_ij^2
    
    # Let's compute using the full matrix for efficiency
    # dist = sqrt(dist_sq)
    dist = np.sqrt(dist_sq)
    # Avoid division by zero
    dist[dist < 1e-12] = 1e-12
    
    # Energy
    # Sum over all i != j
    mask = ~np.eye(N, dtype=bool)
    E = -0.5 * np.sum(np.log(dist[mask]))
    
    # Gradient
    # grad_k = -0.5 * sum_{j != k} (x_k - x_j) / dist_kj^2
    # Let's compute this efficiently
    # For each k, sum over j != k of (x_k - x_j) / dist_kj^2
    # = x_k * sum_{j!=k} 1/dist_kj^2 - sum_{j!=k} x_j / dist_kj^2
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # sum_j 1/dist_kj^2 for each k
    sum_inv_dist_sq = np.sum(inv_dist_sq, axis=1)  # shape (N,)
    
    # sum_j x_j / dist_kj^2 for each k
    # This is X @ inv_dist_sq
    sum_x_inv_dist_sq = X @ inv_dist_sq  # shape (N, 3)
    
    grad = -0.5 * (X * sum_inv_dist_sq[:, np.newaxis] - sum_x_inv_dist_sq)
    
    # Flatten gradient
    grad_flat = grad.flatten()
    
    return E, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin: 118s out of 120s limit
    
    rng = np.random.default_rng(12345)
    
    best_energy = float('inf')
    best_X = None
    
    # Try multiple initial configurations
    # Strategy: Fibonacci sphere with random rotations and perturbations
    n_restarts = 50  # We'll try up to 50 restarts, but stop if time runs out
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial configuration
        if restart == 0:
            # Pure Fibonacci
            X = _fibonacci_sphere(N, rng=None)
        elif restart % 2 == 1:
            # Fibonacci with random rotation
            X = _fibonacci_sphere(N, rng=rng)
        else:
            # Fibonacci with random perturbation
            X = _fibonacci_sphere(N, rng=rng)
            # Add small random perturbation
            noise = rng.standard_normal((N, 3)) * 0.01
            X += noise
            # Normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = X / norms
        
        x0 = X.flatten()
        
        # Run Riemannian L-BFGS-B
        # We'll use standard L-BFGS-B but with a constraint that points stay on sphere
        # Actually, let's use a retraction-based approach
        # For simplicity, we'll use standard L-BFGS-B and then project back to sphere
        # This is not ideal but may work well enough
        
        # Better: Use a custom optimization that respects the manifold
        # Let's use scipy's minimize with a custom objective that projects
        
        # Actually, let's implement a simple Riemannian gradient descent with line search
        # Or use L-BFGS-B on the tangent space
        
        # For now, let's use a simple approach: 
        # 1. Compute gradient
        # 2. Project to tangent space
        # 3. Take step
        # 4. Retract back to sphere
        
        current_X = X.copy()
        current_energy, _ = _energy_and_grad(current_X.flatten())
        
        # Riemannian gradient descent with adaptive step size
        step_size = 0.1
        max_iters = 2000
        tol = 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            # Compute energy and gradient
            E, grad_flat = _energy_and_grad(current_X.flatten())
            grad = grad_flat.reshape(N, 3)
            
            # Project gradient to tangent space
            # grad_tangent = grad - (grad . x) x
            dot = np.sum(grad * current_X, axis=1, keepdims=True)
            grad_tangent = grad - dot * current_X
            
            # Check convergence
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < tol:
                break
            
            # Adaptive step size: start with 0.1, reduce if energy increases
            # Take step in tangent space
            V = -grad_tangent
            V_norms = np.linalg.norm(V, axis=1, keepdims=True)
            V_norms = np.maximum(V_norms, 1e-12)
            V_unit = V / V_norms
            
            # Exponential map
            cos_step = math.cos(step_size)
            sin_step = math.sin(step_size)
            X_trial = cos_step * current_X + sin_step * V_unit
            X_trial = X_trial / np.linalg.norm(X_trial, axis=1, keepdims=True)
            
            E_trial, _ = _energy_and_grad(X_trial.flatten())
            
            if E_trial < E:
                current_X = X_trial
                current_energy = E_trial
                # Increase step size slightly if it worked
                step_size = min(step_size * 1.05, 0.5)
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        # Check if this is better
        if current_energy < best_energy:
            best_energy = current_energy
            best_X = current_X.copy()
        
        # If we've found a very good solution, we can stop early
        if best_energy < -1083.37:
            break
    
    # Final normalization
    if best_X is not None:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        best_X = best_X / norms
    
    return best_X
