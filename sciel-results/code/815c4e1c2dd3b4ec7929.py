"""
Strategy: Riemannian optimization on the Stiefel manifold (orthogonal rows)
using Gradient Descent with Barzilai-Borwein step sizes and periodic
random orthogonal rotations (basin hopping). This differs structurally from
the current spherical-coordinate L-BFGS-B approach by:
1. Working directly with Cartesian coordinates constrained to the sphere via
   projection, avoiding coordinate singularities.
2. Using Barzilai-Borwein adaptive step sizes for faster convergence.
3. Employing random orthogonal rotations as a basin-hopping mechanism, which
   is more effective for symmetric energy landscapes than simple perturbations.
"""
import time
import math
import numpy as np
from scipy.linalg import orth

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the unit sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise squared distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    
    # Ensure no division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # We compute sum over all i != j of -log(d_ij) / 2
    mask = ~np.eye(N, dtype=bool)
    log_dists = -np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = 0.5 * np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: d/dx_i (-log(||x_i - x_j||)) = -(x_i - x_j)/||x_i - x_j||^2
    # Wait, E = -sum log(d). dE/dx_i = -sum_{j != i} (1/d_ij) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j)/d_ij
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # Let's re-verify:
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_i = - sum_{j != i} (1/2) * (1/||x_i - x_j||^2) * 2(x_i - x_j) = - sum_{j != i} (x_i - x_j)/||x_i - x_j||^2
    # The code below computes sum_{j != i} (x_i - x_j)/dist_sq.
    # So grad_x = - (that sum).
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_x[i] = - sum_j (pts[i] - pts[j]) * inv_dist_sq[i,j]
    # diff[i,j] = pts[i] - pts[j]
    # So sum_j diff[i,j] * inv_dist_sq[i,j] is the sum we want, but with a negative sign.
    grad_sum = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    grad = -grad_sum
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _random_rotation_matrix():
    """Generate a random 3x3 orthogonal matrix (rotation)."""
    # Use QR decomposition of a random Gaussian matrix
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q

def _optimize_stiefel(pts_init, time_limit=30.0):
    """
    Optimize points on the sphere using Riemannian Gradient Descent
    with Barzilai-Borwein step sizes and periodic basin hopping.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts_init.copy())
    
    # Barzilai-Borwein state
    prev_pts = pts.copy()
    prev_grad = None
    prev_energy = _energy_and_grad(pts)[0]
    
    # Initial step size
    step_size = 0.01
    min_step = 1e-6
    max_step = 0.1
    
    # Basin hopping parameters
    hop_interval = 200  # Number of gradient steps between hops
    hop_prob = 0.1      # Probability of accepting a worse hop
    current_temp = 1.0  # Temperature for acceptance
    cooling_rate = 0.99
    
    steps_since_hop = 0
    best_energy = prev_energy
    best_pts = pts.copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2.0:
            break
        
        # Compute gradient
        energy, grad = _energy_and_grad(pts)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Barzilai-Borwein step size update
        if prev_grad is not None:
            s = pts - prev_pts
            y = grad - prev_grad
            s_norm_sq = np.sum(s * s)
            y_norm_sq = np.sum(y * y)
            sy = np.sum(s * y)
            
            if s_norm_sq > 1e-12 and y_norm_sq > 1e-12:
                # Alternate between BB1 and BB2
                if steps_since_hop % 2 == 0:
                    # BB1: sy / ||s||^2
                    step_size = sy / s_norm_sq
                else:
                    # BB2: ||s||^2 / sy
                    step_size = s_norm_sq / sy if sy > 1e-12 else step_size
                
                # Clamp step size
                step_size = np.clip(step_size, min_step, max_step)
        
        prev_pts = pts.copy()
        prev_grad = grad.copy()
        
        # Riemannian Gradient Descent step
        # Move in the direction of negative gradient
        # Then project back to the sphere
        new_pts = pts - step_size * grad
        new_pts = _project_to_sphere(new_pts)
        
        # Check for convergence or stagnation
        pts = new_pts
        
        steps_since_hop += 1
        
        # Basin Hopping
        if steps_since_hop >= hop_interval:
            steps_since_hop = 0
            current_temp *= cooling_rate
            
            # Generate a random rotation
            R = _random_rotation_matrix()
            hop_pts = R @ pts.T
            hop_pts = hop_pts.T
            hop_pts = _project_to_sphere(hop_pts)
            
            hop_energy, _ = _energy_and_grad(hop_pts)
            
            # Accept if better or with probability exp(-(delta_E)/T)
            delta_e = hop_energy - best_energy
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(current_temp, 1e-6)):
                pts = hop_pts
                # Reset step size after a hop
                step_size = 0.01
                prev_grad = None
        
        # Early termination if gradient is very small
        if np.linalg.norm(grad) < 1e-10:
            break
    
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    theta = np.arcsin(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    return np.column_stack([x, y, z])

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start strategy
    initializers = [
        lambda: _fibonacci_sphere(N),
        lambda: _random_sphere(N, seed=1),
        lambda: _random_sphere(N, seed=2),
        lambda: _random_sphere(N, seed=3),
        lambda: _random_sphere(N, seed=4),
    ]
    
    # Allocate time to each start
    # First one (Fibonacci) gets more time as it's a good start
    time_allocations = [50, 20, 20, 20, 20]
    
    for i, (init_func, time_alloc) in enumerate(zip(initializers, time_allocations)):
        if time.time() > deadline - 10.0:
            break
        
        try:
            pts_init = init_func()
            remaining = deadline - time.time() - 5.0
            if remaining < 5.0:
                break
            
            actual_time = min(time_alloc, remaining)
            
            pts_opt, energy = _optimize_stiefel(pts_init, time_limit=actual_time)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt.copy()
                
        except Exception as e:
            continue
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            # Run a final short optimization with smaller step sizes
            # We can reuse _optimize_stiefel with a short time limit
            # Or just do a few gradient steps manually
            pts = best_pts
            energy, grad = _energy_and_grad(pts)
            
            # Simple Newton-like or small step GD for polish
            step_size = 0.001
            for _ in range(100):
                if time.time() > deadline - 1.0:
                    break
                new_pts = pts - step_size * grad
                new_pts = _project_to_sphere(new_pts)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    pts = new_pts
                    energy = new_energy
                    grad = new_grad
                else:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
