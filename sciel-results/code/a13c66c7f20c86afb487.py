"""Optimized logarithmic energy minimizer for 150 points on the unit sphere using Riemannian optimization with stereographic projection."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """
    Compute logarithmic energy and Euclidean gradient for points x of shape (N, 3).
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient is the Euclidean gradient in R^3.
    Uses vectorized operations for speed.
    """
    # Compute pairwise differences (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # Compute squared distances (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use the full matrix and take upper triangle
    log_dists = np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)  # Each pair counted twice, so divide by 2
    
    # Gradient: grad[i] = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||
    # This can be computed as: grad = -sum_j (x_i - x_j) / dist_ij
    # = -x_i * sum_j (1/dist_ij) + sum_j x_j / dist_ij
    # Let W_ij = 1/dist_ij for i!=j, 0 for i==j
    # Then grad[i] = -x_i * sum_j W_ij + sum_j W_ij * x_j
    # = -x_i * (W @ 1)_i + (W @ x)_i
    
    W = np.zeros_like(dist)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist  # This sets diagonal to inf, but we zero it out
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij for each i
    row_sums = W.sum(axis=1)  # (N,)
    # W @ x: (N,3) @ (N,3)^T... wait, W is (N,N), x is (N,3)
    # (W @ x) gives (N,3) where [W@x]_i = sum_j W_ij * x_j
    Wx = W @ x  # (N, 3)
    
    grad = Wx - row_sums[:, np.newaxis] * x
    
    return energy, grad


def _log_energy_fast(x):
    """Fast energy computation only (no gradient)."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    log_dists = np.log(dist)
    np.fill_diagonal(log_dists, 0.0)
    return -0.5 * np.sum(log_dists)


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate n points using Halton sequence on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        p = [halton(i, b) for b in [2, 3, 5]]
        theta = 2 * np.pi * p[0]
        phi = np.arccos(2 * p[1] - 1)
        pt = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        norm = np.linalg.norm(pt)
        points.append(pt / norm)
    return np.array(points)


def _normalize(x):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _riemannian_grad_descent(x0, deadline, max_iter=20000, tol=1e-10):
    """
    Riemannian Gradient Descent with adaptive backtracking line search.
    Uses projection retraction onto the sphere.
    """
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    # Adaptive step size
    alpha = 0.1
    alpha_min = 1e-8
    alpha_max = 1.0
    c1 = 1e-4  # Armijo parameter
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
            
        # Direction: negative gradient in tangent space
        step_dir = -grad_tangent
        
        # If gradient is very small, stop
        if grad_norm < 1e-15:
            break
            
        found_alpha = False
        a = alpha
        for _ in range(30):
            if a < alpha_min:
                break
            # Move in tangent direction and retract
            x_trial = x + a * step_dir
            x_trial = _normalize(x_trial)
            
            energy_trial = _log_energy_fast(x_trial)
            
            # Armijo condition: f_new <= f_old - c1 * a * ||grad_tangent||^2
            armijo_rhs = energy - c1 * a * grad_norm ** 2
            if energy_trial <= armijo_rhs:
                found_alpha = True
                break
            a *= 0.5
            
        if not found_alpha:
            # Try very small step
            a = alpha_min
            x_trial = x + a * step_dir
            x_trial = _normalize(x_trial)
            energy_trial = _log_energy_fast(x_trial)
            if energy_trial >= energy:
                break
        
        # Update alpha for next iteration
        if found_alpha:
            # If we used the full alpha, try increasing it
            if a == alpha:
                alpha = min(alpha * 1.2, alpha_max)
            else:
                # We had to backtrack; set alpha to the value that worked
                alpha = a
        else:
            alpha = max(alpha * 0.5, alpha_min)
        
        # Update
        x = x_trial
        energy = energy_trial
        
        # Compute new gradient
        _, grad = _energy_and_grad(x)
        grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
    return x


def _riemannian_cg(x0, deadline, max_iter=5000, tol=1e-10):
    """
    Riemannian Conjugate Gradient method for faster convergence.
    """
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    # Initial direction
    d = -grad_tangent
    beta = 0.0
    alpha = 0.1
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Line search
        found_alpha = False
        a = alpha
        for _ in range(30):
            if a < 1e-10:
                break
            x_trial = x + a * d
            x_trial = _normalize(x_trial)
            energy_trial = _log_energy_fast(x_trial)
            
            if energy_trial < energy:
                found_alpha = True
                break
            a *= 0.5
        
        if not found_alpha:
            break
        
        # Update
        x = x_trial
        energy = energy_trial
        
        # New gradient
        _, grad_new = _energy_and_grad(x)
        grad_new_tangent = grad_new - np.sum(grad_new * x, axis=1, keepdims=True) * x
        
        # Compute beta (Polak-Ribiere)
        grad_diff = np.sum(grad_new_tangent * grad_tangent)
        beta = max(0.0, np.sum(grad_new_tangent * (grad_new_tangent - grad_tangent)) / max(np.sum(grad_tangent * grad_tangent), 1e-30))
        
        # Update direction
        d = -grad_new_tangent + beta * d
        
        # Project direction to tangent space
        d = d - np.sum(d * x, axis=1, keepdims=True) * x
        
        # Reset direction if not making progress
        if np.sum(d * grad_new_tangent) > 0:
            d = -grad_new_tangent
        
        grad_tangent = grad_new_tangent
        alpha = min(alpha * 1.1, 1.0)
    
    return x


def _polish(x0, deadline):
    """
    High-precision polish pass using Riemannian CG with tight tolerance.
    """
    remaining = deadline - time.time()
    if remaining < 2.0:
        return x0
    
    # Run Riemannian CG with tight tolerance
    x_opt = _riemannian_cg(x0, deadline, max_iter=10000, tol=1e-12)
    
    # Validate
    if not np.all(np.isfinite(x_opt)):
        return x0
    if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
        return x0
        
    # Check minimum distance
    diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < (1e-9) ** 2:
        return x0
        
    return x_opt


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("halton", _halton_sphere(N)),
    ]
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(8):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    def _validate_and_update(x, energy):
        nonlocal best_x, best_energy
        if not np.all(np.isfinite(x)):
            return
        if np.min(np.linalg.norm(x, axis=1)) < 1e-15:
            return
        # Check min distance
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        if np.min(dist_sq) < (1e-9) ** 2:
            return
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
    
    # Phase 1: Run Riemannian GD on all initializations
    for name, x0 in initializations:
        if time.time() > deadline - 30.0:
            break
        
        x0 = _normalize(x0)
        
        # Run Riemannian Gradient Descent
        x_opt = _riemannian_grad_descent(x0, deadline - 15.0, max_iter=15000, tol=1e-10)
        
        energy = _log_energy_fast(x_opt)
        _validate_and_update(x_opt, energy)
    
    # Phase 2: Run Riemannian CG on the best result and top initializations
    candidates = []
    if best_x is not None:
        candidates.append(("best", best_x.copy()))
    
    # Also try CG on a few other initializations
    for name, x0 in initializations[:4]:
        if time.time() > deadline - 25.0:
            break
        x0 = _normalize(x0)
        x_gd = _riemannian_grad_descent(x0, deadline - 20.0, max_iter=5000, tol=1e-9)
        energy_gd = _log_energy_fast(x_gd)
        candidates.append((name, x_gd))
    
    for name, x_cand in candidates:
        if time.time() > deadline - 15.0:
            break
        
        x_cg = _riemannian_cg(x_cand, deadline - 10.0, max_iter=5000, tol=1e-11)
        energy_cg = _log_energy_fast(x_cg)
        _validate_and_update(x_cg, energy_cg)
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 5.0:
        x_polished = _polish(best_x, deadline)
        if np.all(np.isfinite(x_polished)):
            energy_polished = _log_energy_fast(x_polished)
            _validate_and_update(x_polished, energy_polished)
    
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    return _normalize(best_x)
