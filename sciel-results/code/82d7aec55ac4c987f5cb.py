"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient w.r.t. free R^3 coordinates.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by zero
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # So grad_i = sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere.
    
    For a point x on the unit sphere, the tangent projection of a vector v is:
    v_tangent = v - (v · x) * x
    """
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _random_sphere(n):
    """Generate n random points on the unit sphere."""
    pts = np.random.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize_from_start(pts0, time_limit=100.0):
    """Optimize points on the sphere using gradient descent with line search.
    
    Uses a simple gradient descent with adaptive step size on the sphere.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    start_time = time.time()
    
    # Initial step size
    step_size = 0.01
    
    # Gradient descent with backtracking line search
    max_iters = 50000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _log_energy_grad(pts)
        grad_tangent = _project_to_tangent(pts, grad)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
        
        # Normalize gradient for direction
        direction = -grad_tangent / (grad_norm + 1e-30)
        
        # Backtracking line search
        alpha = step_size
        success = False
        for _ in range(20):
            # Move on sphere: pts_new = (pts + alpha * direction) normalized
            pts_new = pts + alpha * direction
            pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
            
            new_energy = _log_energy(pts_new)
            if new_energy < energy:
                pts = pts_new
                step_size = min(alpha * 1.5, 0.1)
                success = True
                break
            alpha *= 0.5
        
        if not success:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    pts0 = _make_fibonacci_sphere()
    
    # Try multiple restarts with different starting configurations
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # First optimization from Fibonacci start
    if time.time() < deadline - 10.0:
        pts_opt, energy_opt = _optimize_from_start(pts0, time_limit=60.0)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Try random restarts
    np.random.seed(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random start
        pts_rand = _random_sphere(N)
        
        # Quick optimization
        pts_opt, energy_opt = _optimize_from_start(pts_rand, time_limit=15.0)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Final fine-tuning from best solution
    if time.time() < deadline - 5.0:
        remaining = max(deadline - time.time() - 2.0, 1.0)
        pts_opt, energy_opt = _optimize_from_start(best_pts, time_limit=remaining)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        # Perturb close points
        for _ in range(100):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            # Move point j slightly
            rand_dir = np.random.randn(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
