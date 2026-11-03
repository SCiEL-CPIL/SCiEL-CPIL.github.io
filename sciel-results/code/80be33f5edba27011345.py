"""
Strategy: Spectral Uniformity Optimization
Instead of convex hull facet refinement, we minimize the variance of pairwise distances.
A uniform distribution on the sphere minimizes the sum of squared distances (or maximizes the sum of distances, but variance minimization is smoother).
We use gradient descent on the sphere to minimize the 'clumping' of points, which indirectly minimizes the covering radius by forcing points to spread out evenly.
"""
import numpy as np
import time
import math

N = 130
START_TIME = time.time()
TIME_LIMIT = 115.0 # Safety margin before 120s/150s kill

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _compute_gram(points):
    """Compute Gram matrix G = P P^T."""
    return points @ points.T

def _spectral_energy(points):
    """
    Compute an energy function that encourages uniformity.
    We want to minimize the variance of the off-diagonal entries of the Gram matrix.
    For N points on unit sphere, if they were perfectly uniform (e.g., vertices of a regular polytope),
    the off-diagonal elements would be roughly constant (related to 1/N).
    Minimizing the sum of squares of (G_ij - mean) for i!=j promotes uniformity.
    Actually, minimizing sum_{i!=j} G_ij^2 is equivalent to minimizing the 'clumping'.
    Let's use E = sum_{i!=j} (G_ij)^2.
    Gradient with respect to P:
    E = Tr(G^2) - Tr(G^2_diag)
    dE/dP = 2 G P - 2 G P (where diagonal is zeroed out in gradient? No)
    Let's derive properly.
    E = sum_{i!=j} (p_i . p_j)^2
    Let A be the matrix with 0 on diagonal and 1 elsewhere.
    E = sum_{i,j} A_ij (p_i . p_j)^2
    dE/dp_k = 2 sum_{j!=k} (p_k . p_j) A_kj p_j
             = 2 sum_{j!=k} G_kj p_j
             = 2 (G p_k - G_kk p_k)  (since G_kk=1)
             = 2 (G p_k - p_k)
    Wait, sum_{j!=k} G_kj p_j is the k-th row of G multiplied by P, minus the k-th term G_kk p_k.
    So grad_k = 2 * (G @ p_k - p_k).
    Vectorized: Grad = 2 * (G @ P - P).
    """
    G = _compute_gram(points)
    # Zero out diagonal for energy calculation if needed, but for gradient:
    # grad_k = 2 * sum_{j!=k} G_kj p_j
    # G @ P gives sum_{j} G_kj p_j.
    # We need to subtract the j=k term which is G_kk p_k = 1 * p_k = p_k.
    # So grad = 2 * (G @ P - P).
    
    # To make it more robust against local minima, we can add a small repulsion term 
    # or just rely on the variance. 
    # Actually, minimizing sum G_ij^2 tends to make points orthogonal if possible.
    # For N=130, they can't be orthogonal. It pushes them towards a uniform distribution.
    
    # Let's compute the actual value for monitoring
    off_diag_mask = ~np.eye(N, dtype=bool)
    energy = np.sum(G[off_diag_mask] ** 2)
    
    # Gradient
    grad = 2.0 * (G @ points - points)
    
    return energy, grad

def _gradient_descent(points, time_limit):
    """
    Perform gradient descent on the sphere.
    """
    points = points.copy()
    start_time = time.time()
    
    # Adaptive learning rate
    lr = 0.01
    best_energy = np.inf
    best_points = points.copy()
    no_improve_count = 0
    
    # Momentum
    velocity = np.zeros_like(points)
    beta = 0.9
    
    while time.time() - start_time < time_limit:
        energy, grad = _spectral_energy(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Reduce learning rate if stuck
        if no_improve_count > 50:
            lr *= 0.9
            if lr < 1e-6:
                break
            no_improve_count = 0
            
        # Update velocity
        velocity = beta * velocity - lr * grad
        
        # Project gradient onto tangent space?
        # The update p_new = p + v is not on the sphere.
        # We can just normalize after update.
        # However, the gradient computed is in R^3.
        # To stay on the manifold, we should project the step onto the tangent space.
        # Tangent space at p: {v | p . v = 0}
        # Projected grad: grad_tan = grad - (grad . p) p
        # But since we normalize anyway, the radial component is removed.
        # So simple update + normalize is equivalent to projected gradient descent with line search 1.
        
        points = points + velocity
        
        # Normalize
        points = _normalize(points)
        
        # Check for time
        if time.time() - start_time > time_limit - 2:
            break
            
    return best_points

def _refine_with_annealing(points, time_limit):
    """
    Combine gradient descent with occasional random perturbations (simulated annealing style)
    to escape local minima of the spectral energy.
    """
    points = points.copy()
    start_time = time.time()
    current_temp = 0.05
    min_temp = 0.001
    
    while time.time() - start_time < time_limit - 5:
        # Run a batch of gradient descent steps
        batch_time = min(5.0, time_limit - (time.time() - start_time) - 2)
        if batch_time < 0.5:
            break
            
        points = _gradient_descent(points, batch_time)
        
        # Apply a random perturbation
        if current_temp > min_temp:
            # Random rotation or small random addition
            # Small random addition is easier
            noise = np.random.normal(0, current_temp, points.shape)
            points = _normalize(points + noise)
            
            # Cool down
            current_temp *= 0.95
        else:
            break
            
    return points

def _initial_config():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def construct():
    """
    Entry point.
    1. Initialize with Fibonacci spiral.
    2. Refine using spectral uniformity gradient descent with annealing.
    3. Return the best configuration found.
    """
    start_time = time.time()
    
    # Initial configuration
    points = _initial_config()
    
    # Refine
    # We have about 115 seconds.
    points = _refine_with_annealing(points, TIME_LIMIT)
    
    # Final check and normalization
    points = _normalize(points)
    
    # Ensure no coincident points (should be fine with gradient descent)
    # Check validity
    if not np.all(np.isfinite(points)):
        return _initial_config()
        
    return points

# EVOLVE-BLOCK-START
# (The functions above are the implementation)
# EVOLVE-BLOCK-END
