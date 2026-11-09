"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize, LinearConstraint


def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal (self-distances)
    np.fill_diagonal(dist, 1.0)
    # Only upper triangle for i<j
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _log_energy_and_grad(pts):
    """Compute energy and gradient for log(1/|x_i - x_j|) = -log|xi-xj|.
    
    Energy = sum_{i<j} -log|xi - xj|
    Gradient w.r.t. xi: sum_{j != i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    # diff[i,j] = xi - xj, so diff[i,j]/dist_sq[i,j] gives the vector
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq[dist_sq > 1e-30] = 1.0 / dist_sq[dist_sq > 1e-30]
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_sphere(pts, max_iter=2000, tol=1e-12):
    """Optimize points on the sphere using projected gradient descent with backtracking.
    
    We use the gradient on the tangent space and project back to the sphere.
    """
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    lr = 0.05
    patience = 0
    max_patience = 200
    
    for it in range(max_iter):
        energy, grad = _log_energy_and_grad(pts)
        
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        if abs(energy - best_energy) < tol and it > 100:
            break
        
        # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        
        # Backtracking line search
        accepted = False
        for _ in range(30):
            new_pts = pts - lr * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            new_energy = _log_energy(new_pts)
            
            if new_energy < energy - 1e-12 * max(1.0, abs(energy)):
                pts = new_pts
                accepted = True
                break
            lr *= 0.5
        
        if not accepted:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return _project_to_sphere(best_pts)


def _multi_start_optimize(n_starts=8, max_iter=1500):
    """Run multiple optimization starts from different initial configurations."""
    best_pts = None
    best_energy = float('inf')
    
    for s in range(n_starts):
        if s == 0:
            # Fibonacci spiral
            pts = _fibonacci_sphere(N)
        elif s == 1:
            # Random start
            rng = np.random.default_rng(42 + s)
            pts = rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
        elif s == 2:
            # Slightly perturbed Fibonacci
            rng = np.random.default_rng(100 + s)
            pts = _fibonacci_sphere(N) + 0.01 * rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
        else:
            # Different random seeds
            rng = np.random.default_rng(1000 + s * 7)
            pts = rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
        
        optimized = _optimize_sphere(pts, max_iter=max_iter)
        e = _log_energy(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized.copy()
    
    return _project_to_sphere(best_pts)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Phase 1: Multi-start optimization
    pts = _multi_start_optimize(n_starts=6, max_iter=1000)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Phase 2: Further refinement from the best result
        pts = _optimize_sphere(pts, max_iter=3000, tol=1e-14)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        # Phase 3: One more round of multi-start with perturbation around best
        rng = np.random.default_rng(12345)
        for _ in range(4):
            if time.time() > deadline - 10.0:
                break
            perturbed = pts + 0.005 * rng.standard_normal((N, 3))
            perturbed = _project_to_sphere(perturbed)
            optimized = _optimize_sphere(perturbed, max_iter=1500)
            e_opt = _log_energy(optimized)
            e_cur = _log_energy(pts)
            if e_opt < e_cur:
                pts = optimized
    
    # Final check: ensure all points are valid
    pts = _project_to_sphere(pts)
    
    # Verify no two points are too close
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Add small perturbation to fix
        rng = np.random.default_rng(999)
        pts = pts + 1e-8 * rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
    
    return pts
# EVOLVE-BLOCK-END
