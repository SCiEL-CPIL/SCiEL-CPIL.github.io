"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE_S = 115.0  # safety margin before 120s kill


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    # Compute all pairwise distances
    # Using broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Only use upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip to avoid log(0) or negative
    d = np.maximum(d, 1e-15)
    terms = -np.log(d)
    return math.fsum(terms.tolist())


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _gradient_descent(pts, lr_init=0.01, max_iter=5000, tol=1e-12):
    """
    Gradient descent on the sphere for logarithmic energy minimization.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Project gradient to tangent space: grad_t = grad - (grad . x_i) * x_i
    """
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    lr = lr_init
    
    for it in range(max_iter):
        # Compute pairwise distances and gradients
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # n x n x 3
        dists = np.sqrt(np.sum(diff * diff, axis=2))  # n x n
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Gradient: for each i, sum over j != i of (x_i - x_j) / |d_ij|^2
        # This is equivalent to: (1/d^2) * diff, summed over j for each i
        # But we need to be careful: diff[i,j] = x_i - x_j
        # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        inv_d2 = np.zeros_like(dists)
        mask = dists > 1e-12
        inv_d2[mask] = 1.0 / (dists[mask] ** 2)
        np.fill_diagonal(inv_d2, 0.0)
        
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # n x 3
        
        # Project to tangent space
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_t = grad - dot * pts
        
        # Update with adaptive learning rate
        pts_new = pts - lr * grad_t
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check convergence
        step_norm = np.linalg.norm(pts_new - pts)
        if step_norm < tol:
            pts = pts_new
            break
        
        # Adaptive learning rate: reduce if energy increases
        # (simplified: just decay lr)
        if it % 100 == 0 and it > 0:
            lr *= 0.95
        
        pts = pts_new
    
    return pts


def _repulsion_force(pts):
    """
    Alternative: use a force-based approach with log repulsion.
    Force on i from j: (x_i - x_j) / |x_i - x_j|^2 (same as gradient above)
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    force = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return force


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    def remaining():
        return DEADLINE_S - (time.time() - t0)
    
    # Multiple restarts with random perturbations + gradient descent
    n_restarts = 30
    max_iter_per_gd = 3000
    
    for r in range(n_restarts):
        if remaining() < 15.0:
            break
        
        # Create perturbed starting point
        if r == 0:
            start = best_pts.copy()
        elif r < 10:
            # Small perturbations of best
            start = best_pts + 0.05 * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        elif r < 20:
            # Medium perturbations
            start = best_pts + 0.2 * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        else:
            # Random restarts
            start = _random_sphere(N, rng)
        
        # Gradient descent
        gd_pts = _gradient_descent(start, lr_init=0.005, max_iter=max_iter_per_gd, tol=1e-14)
        
        e = _log_energy(gd_pts)
        if e < best_energy:
            best_energy = e
            best_pts = gd_pts
    
    # Final polishing: more iterations on the best
    if remaining() > 10.0:
        polished = _gradient_descent(best_pts, lr_init=0.001, max_iter=10000, tol=1e-15)
        e = _log_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pts = polished
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Validate
    assert best_pts.shape == (N, 3), f"Wrong shape: {best_pts.shape}"
    assert np.all(np.isfinite(best_pts)), "Non-finite values"
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_d = dists[iu].min()
    assert min_d > 1e-9, f"Points too close: {min_d}"
    
    return best_pts
