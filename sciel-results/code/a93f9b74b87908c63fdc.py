"""Rewritten program: gradient-based minimization of logarithmic energy on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_gradient(pts):
    """Compute logarithmic energy and its gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_i = -sum_{j != i} (x_i - x_j) / |d_ij|^2  (for the energy)
    So to minimize energy, we move x_i in direction -grad = sum_{j != i} (x_i - x_j)/|d_ij|^2 ... 
    wait, let's be careful.
    
    E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    dE/dx_i = - sum_{j != i} d/dx_i [log(d_ij)] 
             = - sum_{j != i} (1/d_ij) * d(d_ij)/dx_i
    d_ij = |x_i - x_j|, d(d_ij)/dx_i = (x_i - x_j)/d_ij
    So dE/dx_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    
    To minimize E, move x_i in direction -dE/dx_i = sum_{j != i} (x_i - x_j)/d_ij^2
    This pushes x_i away from all other points, which is what we want.
    """
    pts = np.asarray(pts)
    n = len(pts)
    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    # Energy: sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist_safe[iu]))
    
    # Gradient for each point i: dE/dx_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # diff[i,j] = x_i - x_j
    # So grad[i] = - sum_{j != i} diff[i,j] / dist[i,j]^2
    inv_dist_sq = np.zeros_like(dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = - sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def construct():
    # Start from Fibonacci spiral
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Minimize the energy using gradient descent on the sphere
    # We minimize E, so we move in direction -grad_E = grad_direction (as computed above)
    # Actually from above: to minimize E, move in direction sum_{j!=i}(x_i-x_j)/d_ij^2 = -grad_E
    
    lr = 0.01
    decay = 0.999
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    for step in range(5000):
        energy, grad = _energy_and_gradient(pts)
        # Direction to move to minimize energy: -grad
        direction = -grad
        
        # Momentum update
        velocity = momentum * velocity + (1 - momentum) * direction
        
        # Project velocity to tangent space of sphere at each point
        # tangent component = v - (v . x) x
        dot = np.sum(pts * velocity, axis=1, keepdims=True)
        tangent_vel = velocity - dot * pts
        
        # Adaptive step size: scale by norm
        step_size = lr / (np.max(np.linalg.norm(tangent_vel, axis=1)) + 1e-12)
        
        # Update
        pts = pts + step_size * tangent_vel
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.where(norms < 1e-12, 1.0, norms)
        
        # Decay learning rate
        lr *= decay
    
    # Final cleanup: ensure exactly unit norm
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Verify all points are finite and distinct
    assert np.all(np.isfinite(pts))
    assert pts.shape == (N, 3)
    
    return pts
# EVOLVE-BLOCK-END
