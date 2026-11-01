"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """Compute logarithmic energy and gradient for points x of shape (N, 3).
    
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient: dE/dx_k = sum_{j != k} (x_j - x_k) / ||x_j - x_k||
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid log(0) and division by zero
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq_safe)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient computation using vectorized approach
    # For each pair (i,j): contribution to grad[i] is -(x_i - x_j)/dist, to grad[j] is (x_i - x_j)/dist
    # Equivalently: grad[k] = sum_{j!=k} (x_j - x_k)/dist(k,j)
    # Build weight matrix: W[i,j] = 1/dist(i,j) for i!=j, 0 for i==j
    W = np.zeros_like(dist)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist  # (N, N)
    np.fill_diagonal(W, 0.0)
    
    # grad[k] = sum_j W[k,j] * (x_j - x_k) = (W @ x)[k] - W[k].sum() * x[k]
    Wx = W @ x  # (N, 3)
    W_row_sum = W.sum(axis=1, keepdims=True)  # (N, 1)
    grad = Wx - W_row_sum * x
    
    return energy, grad


def _log_energy_fast(x):
    """Fast energy computation only (no gradient)."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


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


def _lbfgs_optimize(x0, deadline, maxiter=5000, gtol=1e-12):
    """Run L-BFGS optimization with deadline guard."""
    try:
        def _sphere_energy_grad(flat_x):
            x = flat_x.reshape(N, 3)
            x = _normalize(x)
            e, g = _energy_and_grad(x)
            # Project gradient to tangent space
            g_tangent = g - np.sum(g * x, axis=1, keepdims=True) * x
            return e, g_tangent.flatten()
        
        result = minimize(
            _sphere_energy_grad,
            x0.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': 1e-14,
                'disp': False,
                'maxls': 50
            }
        )
        return _normalize(result.x.reshape(N, 3))
    except Exception:
        return x0


def _polish(x0, deadline):
    """High-precision polish pass with very tight tolerances."""
    remaining = deadline - time.time()
    if remaining < 5.0:
        return x0
    
    x = x0.copy()
    
    # Phase 1: L-BFGS with tight tolerance
    try:
        def _sphere_energy_grad_tight(flat_x):
            x_local = flat_x.reshape(N, 3)
            x_local = _normalize(x_local)
            e, g = _energy_and_grad(x_local)
            g_tangent = g - np.sum(g * x_local, axis=1, keepdims=True) * x_local
            return e, g_tangent.flatten()
        
        result = minimize(
            _sphere_energy_grad_tight,
            x.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 10000,
                'gtol': 1e-14,
                'ftol': 1e-16,
                'disp': False,
                'maxls': 100
            }
        )
        x = _normalize(result.x.reshape(N, 3))
    except Exception:
        pass
    
    # Phase 2: Riemannian CG for final polish
    if time.time() < deadline - 2.0:
        x = _riemannian_cg(x, deadline - 2.0, max_iter=500)
    
    # Validate
    if not np.all(np.isfinite(x)):
        return x0
    if np.any(np.linalg.norm(x, axis=1) < 1e-15):
        return x0
    
    # Check minimum distance
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < (1e-9) ** 2:
        return x0
    
    return x


def _grad_descent(x0, deadline, lr=0.01, max_iter=5000):
    """Simple gradient descent with adaptive learning rate as fallback."""
    x = x0.copy()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        energy, grad = _energy_and_grad(x)
        
        # Project gradient to tangent space
        x_flat = x
        grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        # Adaptive step size
        step = lr / (1.0 + 0.1 * it)
        x_new = x - step * grad_tangent
        x_new = _normalize(x_new)
        
        # Check if energy decreased
        energy_new = _log_energy_fast(x_new)
        if energy_new < energy:
            x = x_new
            lr *= 1.05
        else:
            lr *= 0.7
            if lr < 1e-10:
                break
    
    return x


def _riemannian_cg(x0, deadline, max_iter=2000):
    """Riemannian conjugate gradient method for sphere optimization."""
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Project gradient to tangent space
    grad_tangent = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    d = -grad_tangent.copy()
    d = d - np.sum(d * x, axis=1, keepdims=True) * x  # Ensure tangent
    
    prev_grad_tangent = grad_tangent.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Armijo backtracking line search
        alpha = 0.5
        c1 = 1e-4
        grad_d = np.sum(grad_tangent * d)
        
        accepted = False
        for _ in range(40):
            # Retraction: project to sphere
            x_trial = x + alpha * d
            x_trial = _normalize(x_trial)
            
            energy_trial = _log_energy_fast(x_trial)
            
            # Armijo condition
            if energy_trial <= energy + c1 * alpha * grad_d:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try very small step
            alpha = 1e-6
            x_trial = x + alpha * d
            x_trial = _normalize(x_trial)
            energy_trial = _log_energy_fast(x_trial)
            if energy_trial >= energy:
                break
        
        # Update
        x = x_trial
        energy = energy_trial
        
        # New gradient
        _, grad_new = _energy_and_grad(x)
        grad_new_tangent = grad_new - np.sum(grad_new * x, axis=1, keepdims=True) * x
        
        # Check convergence
        if np.linalg.norm(grad_new_tangent) < 1e-11:
            break
        
        # Polak-Ribiere beta
        diff_grad = grad_new_tangent - prev_grad_tangent
        denom = np.sum(prev_grad_tangent * prev_grad_tangent) + 1e-15
        beta = np.sum(grad_new_tangent * diff_grad) / denom
        beta = max(0.0, beta)
        
        d_new = -grad_new_tangent + beta * d
        d_new = d_new - np.sum(d_new * x, axis=1, keepdims=True) * x
        
        # Restart if not descent direction
        if np.sum(d_new * grad_new_tangent) > 0:
            d_new = -grad_new_tangent.copy()
            d_new = d_new - np.sum(d_new * x, axis=1, keepdims=True) * x
        
        d = d_new
        prev_grad_tangent = grad_new_tangent
        grad_tangent = grad_new_tangent
    
    return x


def _is_valid(x):
    """Check if a point configuration is valid."""
    if x is None or not np.all(np.isfinite(x)):
        return False
    if x.shape != (N, 3):
        return False
    if np.any(np.linalg.norm(x, axis=1) < 1e-15):
        return False
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    if np.min(dist_sq) < (1e-9) ** 2:
        return False
    return True


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
    
    # Add random initializations with different seeds
    for seed in [42, 123, 7]:
        np.random.seed(seed)
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    # Add a jittered fibonacci initialization
    np.random.seed(99)
    x_fib_jitter = _fibonacci_sphere(N) + 0.01 * np.random.randn(N, 3)
    x_fib_jitter = _normalize(x_fib_jitter)
    initializations.append(("fib_jitter", x_fib_jitter))
    
    for name, x0 in initializations:
        if time.time() > deadline - 10.0:
            break
        
        x0 = _normalize(x0)
        
        # Strategy 1: L-BFGS
        x_opt = x0
        if time.time() < deadline - 20.0:
            x_opt = _lbfgs_optimize(x0, deadline)
            if _is_valid(x_opt):
                energy = _log_energy_fast(x_opt)
                if energy < best_energy:
                    best_energy = energy
                    best_x = x_opt.copy()
        
        # Strategy 2: Riemannian CG from L-BFGS result
        if time.time() < deadline - 10.0:
            start_cg = x_opt if _is_valid(x_opt) else x0
            x_cg = _riemannian_cg(start_cg, deadline)
            if _is_valid(x_cg):
                energy_cg = _log_energy_fast(x_cg)
                if energy_cg < best_energy:
                    best_energy = energy_cg
                    best_x = x_cg.copy()
    
    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 5.0:
        x_polished = _polish(best_x, deadline)
        if _is_valid(x_polished):
            energy_polished = _log_energy_fast(x_polished)
            if energy_polished < best_energy:
                best_energy = energy_polished
                best_x = x_polished.copy()
    
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    return _normalize(best_x)
