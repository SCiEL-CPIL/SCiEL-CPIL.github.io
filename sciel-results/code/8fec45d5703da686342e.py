"""
Combined program: Multi-start Riemannian Optimized Logarithmic Energy Minimization
for 150 points on the unit sphere.

Strategy:
1. Generate multiple diverse initial configurations (Fibonacci, Halton, Random).
2. For each configuration, run a Riemannian Gradient Descent with Momentum 
   (Nesterov acceleration) using the exponential map on the sphere manifold.
   This is more efficient and geometrically consistent than Simulated Annealing 
   for refining local minima.
3. Use a backtracking line search to ensure robust convergence.
4. Optionally, polish the best result with L-BFGS-B if time permits.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 145.0  # Safety margin for 150s limit


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    Vectorized implementation.
    """
    n = pts.shape[0]
    
    # Compute pairwise differences and squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient calculation:
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    weighted_sum_pts = inv_d2 @ pts  # (n, 3)
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (n,)
    grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space: g_tangent = g - (g . x) * x
    dot_prod = np.sum(grad * pts, axis=1)  # (n,)
    grad = grad - dot_prod[:, np.newaxis] * pts
    
    # Energy calculation: sum_{i<j} log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed (no gradient)."""
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """
    Apply exponential map on sphere: move points along tangent vectors.
    x_new = cos(||v||) * x + sin(||v||)/||v|| * v
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Handle zero norms to avoid division by zero
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    cos_norms = np.cos(safe_norms)
    sin_norms_over_norms = np.sin(safe_norms) / safe_norms
    # For zero norms, the term sin/||v|| * v becomes 0, so we mask it
    mask = norms > 1e-15
    result = cos_norms * pts
    result[mask] += sin_norms_over_norms[mask] * tangent_vecs[mask]
    return result


def _riemannian_optimize(pts, time_limit=20.0, max_iter=500, momentum=0.9):
    """
    Riemannian Gradient Descent with Nesterov Momentum on the sphere.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, current_grad = _energy_and_grad(current_pts)
    
    # Momentum velocity
    v = np.zeros_like(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Initial step size estimate
    step_size = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 0.1:
            break
            
        # Nesterov step: look-ahead
        # v_new = momentum * v - step_size * grad
        # x_new = exp_x(v_new)
        
        # We use a simple momentum update on the tangent space
        v = momentum * v - step_size * current_grad
        
        # Backtracking line search
        alpha = 1.0
        accepted = False
        best_local_pts = current_pts
        best_local_energy = current_energy
        best_local_grad = current_grad
        
        for ls in range(30):
            # Update using exponential map
            proposed_pts = _exponential_map(current_pts, alpha * v)
            proposed_energy, proposed_grad = _energy_and_grad(proposed_pts)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c * alpha * grad^T d
            # Here d is in tangent space. The directional derivative is grad^T v.
            # Since we are on a manifold, we approximate the condition as:
            # proposed_energy < current_energy - 0.01 * alpha * np.sum(current_grad * v)
            
            # Simpler robust condition: just check for improvement
            if proposed_energy < best_local_energy:
                best_local_pts = proposed_pts
                best_local_energy = proposed_energy
                best_local_grad = proposed_grad
                accepted = True
            else:
                # If no improvement, shrink alpha
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        if not accepted:
            # If no step was accepted, try a smaller base step size
            step_size *= 0.5
            if step_size < 1e-12:
                break
            continue
            
        # Update state
        current_pts = best_local_pts
        current_energy = best_local_energy
        current_grad = best_local_grad
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # Check convergence
        if np.linalg.norm(current_grad) < 1e-10:
            break
            
        # Adaptive step size increase if we made good progress
        if it > 0 and current_energy < best_energy - 1e-6:
            step_size = min(step_size * 1.05, 0.1)
            
    return best_pts, best_energy


def _lbfgs_optimize(pts, time_limit=10.0):
    """
    Optimize using L-BFGS-B with projected gradient for final polish.
    """
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 2000, 'maxls': 50})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts, _log_energy_fast(pts)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere
    hal_pts = _halton_sphere(N)
    initializations.append(hal_pts.copy())
    
    # 3. Random Uniform (4 seeds)
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
        
    # 4. Perturbed Fibonacci (3 seeds)
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.05
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
        
    # Optimize each initialization
    # We have ~145s. Let's allocate time dynamically.
    # 8 initializations. ~15s each for Riemannian Opt.
    # Then final polish.
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:  # Leave 15s for final polish
            break
            
        # Phase 1: Riemannian Optimization
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        time_per_init = min(15.0, remaining / (len(initializations) - idx))
        
        try:
            pts_opt, e_opt = _riemannian_optimize(init_pts, time_limit=time_per_init)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass

    # Phase 2: Final Polish with L-BFGS-B if time allows
    if best_pts is not None:
        time_left = deadline - time.time()
        if time_left > 5.0:
            try:
                pts_polished, e_polished = _lbfgs_optimize(best_pts, time_limit=min(time_left - 2, 10.0))
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_pts = pts_polished.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = fib_pts
    
    # Final validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                rng = np.random.default_rng(i+j)
                best_pts[j] += 1e-8 * rng.randn(3)
                best_pts[j] = _normalize(best_pts[j])
                
    return best_pts
