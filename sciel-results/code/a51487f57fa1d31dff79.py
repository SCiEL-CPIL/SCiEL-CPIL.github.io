"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import lpmv

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)."""
    n = len(points)
    # Compute all pairwise distances
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid self-pairs and double counting
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _energy_and_grad(points):
    """Compute energy and gradient of the logarithmic energy.
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = ||x_i - x_j||
    dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2  * (-1) ... 
    Actually: d/dx_i [-log(||x_i - x_j||)] = -(x_i - x_j)/||x_i - x_j||^2
    So gradient for point i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    But we need to project onto the tangent plane since points are constrained to sphere.
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    E = float(np.sum(np.log(1.0 / d)))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # This is the unconstrained gradient; we project to tangent plane
    # grad_i = sum_j (x_i - x_j) / dist_ij^2
    # Note: 1/dist_ij^2 * (x_i - x_j)
    inv_dist_sq = 1.0 / dist_sq  # shape (n, n)
    inv_dist_sq = np.where(np.eye(n, dtype=bool), 0.0, inv_dist_sq)
    # grad[i] = sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = sum_j inv_dist_sq[i,j] * x_i - sum_j inv_dist_sq[i,j] * x_j
    # = x_i * sum_j inv_dist_sq[i,j] - sum_j inv_dist_sq[i,j] * x_j
    row_sum = np.sum(inv_dist_sq, axis=1)  # shape (n,)
    grad = points * row_sum[:, np.newaxis] - inv_dist_sq @ points  # shape (n, 3)
    
    # Project gradient onto tangent plane: grad_tangent = grad - (grad . x_i) x_i
    dot = np.sum(grad * points, axis=1)  # shape (n,)
    grad_tangent = grad - dot[:, np.newaxis] * points
    
    return E, grad_tangent


def _spherical_harmonic_basis(L_max, n_points):
    """Generate spherical harmonic basis vectors for each point.
    
    Returns a list of basis functions: each is an array of shape (n_points, 3)
    representing the gradient of the spherical harmonic on the sphere.
    """
    # For simplicity and speed, use a truncated set of low-order harmonics
    # Y_l^m for l=0..L_max, m=-l..l
    # The gradient of Y_l^m on the sphere gives a tangent vector field
    
    # Actually, for the spectral flow method, we parameterize:
    # x_i(t) = normalize(x_i(0) + sum_{k} c_k * v_k(x_i))
    # where v_k are basis vector fields on the sphere.
    
    # Simpler approach: use random tangent vectors as basis, or use
    # the eigenvectors of the Laplacian on the point cloud.
    
    # For now, let's use a different approach: direct gradient descent with
    # adaptive step size and multiple restarts.
    
    pass


def _gradient_descent(points, max_time, lr_init=0.01):
    """Run gradient descent with adaptive learning rate."""
    start_time = time.time()
    points = points.copy()
    E, grad = _energy_and_grad(points)
    best_E = E
    best_points = points.copy()
    
    lr = lr_init
    patience = 0
    patience_max = 500
    
    while time.time() - start_time < max_time:
        # Normalize points to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        points = points / norms
        
        E, grad = _energy_and_grad(points)
        
        if E < best_E:
            best_E = E
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Update
        points = points - lr * grad
        
        # Decay learning rate
        lr *= 0.999
        if patience > patience_max:
            # Reduce learning rate significantly and reset patience
            lr *= 0.5
            patience = 0
            if lr < 1e-12:
                break
    
    return best_points, best_E


def _newton_like(points, max_time):
    """A more sophisticated optimization using natural gradient / Riemannian flow."""
    start_time = time.time()
    points = points.copy()
    E, grad = _energy_and_grad(points)
    best_E = E
    best_points = points.copy()
    
    lr = 0.005
    momentum = 0.9
    vel = np.zeros_like(points)
    
    while time.time() - start_time < max_time:
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        points = points / norms
        
        E, grad = _energy_and_grad(points)
        
        # Momentum update
        vel = momentum * vel + (1 - momentum) * grad
        points = points - lr * vel
        
        if E < best_E:
            best_E = E
            best_points = points.copy()
        
        # Adaptive lr
        if np.linalg.norm(grad) < 1e-12:
            break
        lr *= 0.9995
    
    return best_points, best_E


def construct():
    deadline = time.time() + 140  # 140s out of 150s to be safe
    
    # Start from Fibonacci spiral
    points0 = _fibonacci_sphere(N)
    
    # Run gradient descent with momentum
    best_points, best_E = _newton_like(points0, max_time=100)
    
    remaining = deadline - time.time()
    if remaining > 5:
        # Try a second phase with smaller steps
        # Also try from a slightly perturbed initial condition
        rng = np.random.RandomState(42)
        points1 = points0 + rng.normal(0, 0.01, size=points0.shape)
        # Normalize
        norms = np.linalg.norm(points1, axis=1, keepdims=True)
        points1 = points1 / norms
        
        points2, E2 = _newton_like(points1, max_time=min(remaining - 5, 30))
        if E2 < best_E:
            best_points, best_E = points2, E2
    
    remaining = deadline - time.time()
    if remaining > 5:
        # Final polish with very small steps
        points3 = best_points.copy()
        E3, grad3 = _energy_and_grad(points3)
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            norms = np.linalg.norm(points3, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            points3 = points3 / norms
            E3, grad3 = _energy_and_grad(points3)
            if np.linalg.norm(grad3) < 1e-14:
                break
            points3 = points3 - 0.001 * grad3
            if E3 < best_E:
                best_E = E3
                best_points = points3.copy()
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
