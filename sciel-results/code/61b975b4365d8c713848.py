"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _fibonacci_points(n, seed_offset=0.0):
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    # Pairwise distances
    # Use a vectorized approach for speed
    # pts shape (N, 3)
    # Compute all pairwise squared distances
    # To save memory and time for N=150, we can do it in chunks or just directly
    # 150x150 is small.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # We only want i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    # Clip to avoid log of very small numbers if any points are too close
    pair_dists = np.clip(pair_dists, 1e-15, None)
    return -np.sum(np.log(pair_dists))


def _log_energy_grad(pts):
    """Compute energy and gradient.
    E = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2  ... wait.
    Let's derive:
    E = - sum_{i<j} ln(||x_i - x_j||)
    dE/dx_i = - sum_{j != i} (1/||x_i - x_j||) * d(||x_i - x_j||)/dx_i
    d(||x_i - x_j||)/dx_i = (x_i - x_j) / ||x_i - x_j||
    So dE/dx_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # Gradient of distance: diff / dist
    # We need sum over j of (x_i - x_j) / d_ij^2
    # This is sum over j of (diff[i,j] / d_ij^2)
    # Note: diff[i,j] = x_i - x_j
    # So grad[i] = sum_j (x_i - x_j) / d_ij^2
    # Which is - sum_j (x_j - x_i) / d_ij^2
    
    # Let's compute the matrix of 1/d^2
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad = diff @ inv_d2? 
    # diff[i,j,k] = x_i[k] - x_j[k]
    # We want sum_j diff[i,j,k] * inv_d2[i,j]
    # This is (diff * inv_d2[:, :, np.newaxis]) .sum(axis=1)
    grad = (diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    
    # The energy gradient is -grad?
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    # So grad_E = - (sum (x_i - x_j)/d^2)
    # My `grad` variable above is sum (x_i - x_j)/d^2.
    # So dE/dx = -grad.
    
    # Wait, let's re-verify signs.
    # E = - sum log d
    # dE/dx = - sum (1/d) * (1/d) * (x_i - x_j)  ... no
    # d/dx log(||x||) = x/||x||^2
    # d/dx_i (-log||x_i - x_j||) = -(x_i - x_j)/||x_i - x_j||^2
    # Sum over j.
    # So Gradient of E w.r.t x_i is - sum_j (x_i - x_j)/d_ij^2.
    
    grad_E = -grad
    
    energy = _log_energy(pts)
    return energy, grad_E


def _spherical_laplace(pts, sigma=0.05):
    """Perturb points on the sphere."""
    noise = np.random.normal(0, sigma, size=pts.shape)
    new_pts = pts + noise
    return _normalize(new_pts)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    best_pts = _fibonacci_points(N)
    best_energy = _log_energy(best_pts)
    
    # Try a few different initializations
    # 1. Standard Fibonacci
    # 2. Fibonacci with slight rotation
    # 3. Random
    
    inits = []
    inits.append(best_pts)
    
    # Generate a few random inits
    for _ in range(5):
        random_pts = np.random.normal(0, 1, size=(N, 3))
        inits.append(_normalize(random_pts))
    
    # Also try a "rings" initialization or other structures if time permits
    # For now, stick with Fibonacci and Randoms.
    
    current_energy = best_energy
    current_pts = best_pts
    
    # Optimization loop
    # Use L-BFGS-B? No, it's a sphere constraint.
    # We can use projected gradient descent or use scipy minimize with a penalty or re-projection.
    # A common technique: parameterize by angles? Hard for 3D.
    # Better: use gradient descent with re-projection.
    # Or use scipy.optimize.minimize with method='L-BFGS-B' on flattened coords and add penalty for norm != 1?
    # Actually, re-projecting after each step is robust.
    
    # Let's implement a simple gradient descent with momentum and re-projection.
    # Or use scipy minimize on the sphere via a wrapper that projects?
    # scipy.optimize.minimize doesn't handle manifold constraints natively well without specific methods.
    # Let's use a custom gradient descent with adaptive step size (line search).
    
    def optimize_from(pts0, max_time=5.0):
        pts = pts0.copy()
        lr = 0.05
        momentum = 0.9
        v = np.zeros_like(pts)
        
        # Line search
        def line_search(pts, grad, lr):
            # Backtracking line search
            energy0 = _log_energy(pts)
            step = lr
            for _ in range(20):
                new_pts = _normalize(pts - step * grad)
                new_energy = _log_energy(new_pts)
                if new_energy < energy0 - 1e-4 * step * np.sum(grad * grad):
                    return new_pts, new_energy, step
                step *= 0.5
            return pts, energy0, 0.0
        
        t_start = time.time()
        while time.time() - t_start < max_time:
            energy, grad = _log_energy_grad(pts)
            
            # Update velocity
            v = momentum * v - lr * grad
            new_pts, new_energy, step_used = line_search(pts, v, 1.0)
            
            if step_used == 0.0:
                # Failed to improve, reduce lr
                lr *= 0.5
                if lr < 1e-6:
                    break
            else:
                pts = new_pts
                energy = new_energy
                # Maybe increase lr slightly if progress was good?
                # Keep it simple for now.
                
        return pts, energy
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Optimize this init
        try:
            opt_pts, opt_energy = optimize_from(init, max_time=10.0)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
            
    # If we have time left, do a polish pass on the best point with higher precision
    if time.time() < deadline - 10.0:
        try:
            # Use a finer optimization
            # Maybe run optimize_from again with smaller lr and longer time
            opt_pts, opt_energy = optimize_from(best_pts, max_time=deadline - time.time() - 5.0)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass

    # Final check for validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    # Ensure no points are too close
    # (The optimizer should prevent this, but just in case)
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
