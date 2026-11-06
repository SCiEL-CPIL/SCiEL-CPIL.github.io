"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)  # Avoid division by zero on diagonal
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} -log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: for each pair (i,j), contribution to x_i is -(x_i-x_j)/d_ij^2
    # grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # Using the diff matrix: diff[i,j] = x_i - x_j
    # grad[i] = -sum_{j!=i} diff[i,j] / d2[i,j]
    
    # Create weight matrix: 1/d2 for i!=j, 0 for i==j
    weights = np.zeros_like(d2)
    np.fill_diagonal(weights, 0)
    weights = 1.0 / d2
    
    # grad[i] = -sum_j weights[i,j] * diff[i,j]
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space (remove radial component)
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_from_start(pts0, deadline):
    """Optimize from a starting configuration using L-BFGS-B on the sphere."""
    # Flatten to 1D for scipy
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e
    
    # We'll use gradient descent with momentum instead, as L-BFGS-B doesn't handle
    # the sphere constraint well. Let's use a custom optimizer.
    
    pts = pts0.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Gradient descent with adaptive step size and momentum
    v = np.zeros_like(pts)  # momentum
    lr = 0.01
    momentum = 0.9
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Update with momentum
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        # Check if step improved
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.05, 0.1)  # Increase learning rate slightly
        else:
            lr *= 0.7
            v *= 0.5  # Reduce momentum
            if lr < 1e-10:
                break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple starting configurations
    best_pts = None
    best_energy = np.inf
    
    # Start 1: Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    pts_opt, e_opt = _optimize_from_start(pts0, deadline)
    if e_opt < best_energy:
        best_energy = e_opt
        best_pts = pts_opt
    
    # Start 2: Random points (seeded for reproducibility)
    if time.time() < deadline - 10:
        pts0 = _random_sphere(N, seed=42)
        pts_opt, e_opt = _optimize_from_start(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Start 3: Another random
    if time.time() < deadline - 10:
        pts0 = _random_sphere(N, seed=123)
        pts_opt, e_opt = _optimize_from_start(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final refinement with smaller steps
    if best_pts is not None and time.time() < deadline - 5:
        # Do a final pass with very small learning rate
        pts = best_pts.copy()
        lr = 0.001
        v = np.zeros_like(pts)
        for it in range(2000):
            if time.time() > deadline:
                break
            e, grad = _log_energy_grad(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            v = 0.9 * v - lr * grad
            pts_new = pts + v
            pts_new = _project_to_sphere(pts_new)
            e_new = _log_energy(pts_new)
            if e_new < e:
                pts = pts_new
            else:
                lr *= 0.5
                v *= 0.5
    
    return best_pts
# EVOLVE-BLOCK-END
