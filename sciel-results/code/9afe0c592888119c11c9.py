"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n uniformly random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.randn(n, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Compute energy and gradient for N points on the sphere.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, gradient) where gradient is (N, 3).
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid log(0) on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / (dists_sq)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _energy(x):
    """Compute just the energy for a set of points."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    return -math.fsum(log_dists.tolist())


def _tangent_project(grad, x):
    """Project gradient to tangent space at each point."""
    radial = np.sum(grad * x, axis=1, keepdims=True)
    return grad - radial * x


def _exp_map(x, v):
    """Riemannian exponential map on the unit sphere.
    x: point on sphere (N, 3)
    v: tangent vector at x (N, 3)
    Returns new point on sphere.
    """
    # Ensure v is truly tangent
    radial_v = np.sum(v * x, axis=1, keepdims=True)
    v = v - radial_v * x
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    safe_theta = np.where(theta > 1e-15, theta, 1.0)
    sin_theta_over_theta = np.sin(safe_theta) / safe_theta
    cos_theta = np.cos(safe_theta)
    # Handle small theta case
    sin_theta_over_theta = np.where(theta > 1e-15, sin_theta_over_theta, 1.0)
    cos_theta = np.where(theta > 1e-15, cos_theta, 1.0)
    result = cos_theta * x + sin_theta_over_theta * v
    # Normalize to ensure on sphere
    result_norms = np.linalg.norm(result, axis=1, keepdims=True)
    result_norms = np.maximum(result_norms, 1e-15)
    return result / result_norms


def _riemannian_cg_bb(x0, max_iter=5000, tol=1e-14, deadline=None):
    """Riemannian Conjugate Gradient with Barzilai-Borwein step sizes.
    
    This operates entirely in tangent space using exponential map updates,
    avoiding Euclidean projection drift. BB step sizes adapt to local curvature.
    """
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    e, g = _energy_and_grad(x)
    grad_t = _tangent_project(g, x)
    
    # Initial BB step size
    alpha = 0.01
    d = -grad_t.copy()
    
    # Previous gradient and step for BB computation
    prev_grad_t = None
    prev_step = None
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Line search with BB step size
        alpha_ls = alpha
        accepted = False
        for ls_iter in range(40):
            x_trial = _exp_map(x, alpha_ls * d)
            e_trial, g_trial = _energy_and_grad(x_trial)
            grad_t_trial = _tangent_project(g_trial, x_trial)
            
            if e_trial < best_e - 1e-16 or (e_trial < e - 1e-16 and ls_iter > 5):
                # Accept step
                x = x_trial
                e = e_trial
                if e < best_e:
                    best_x = x.copy()
                    best_e = e
                grad_t = grad_t_trial
                accepted = True
                break
            else:
                alpha_ls *= 0.5
                if alpha_ls < 1e-18:
                    break
        
        if not accepted:
            # Reset direction if no step accepted
            d = -grad_t.copy()
            prev_grad_t = None
            prev_step = None
            continue
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
        
        # Compute BB step size for next iteration
        if prev_grad_t is not None and prev_step is not None:
            grad_diff = grad_t - prev_grad_t
            step_diff = x - prev_step  # This is approximate; in Riemannian setting, use parallel transport
            
            # For simplicity, compute in Euclidean space as approximation
            num_s = np.sum(grad_diff * d)
            num_y = np.sum(grad_diff * (grad_t - prev_grad_t))
            
            # Barzilai-Borwein step sizes
            if abs(num_s) > 1e-20:
                alpha_s = num_s / (np.sum(d * d) + 1e-20)
            else:
                alpha_s = alpha
            
            if abs(num_y) > 1e-20:
                alpha_y = num_y / (np.sum((grad_t - prev_grad_t) * (grad_t - prev_grad_t)) + 1e-20)
            else:
                alpha_y = alpha
            
            # Alternate between alpha_s and alpha_y
            if it % 2 == 0:
                alpha = np.clip(alpha_s, 1e-12, 0.5)
            else:
                alpha = np.clip(alpha_y, 1e-12, 0.5)
        
        # Update conjugate direction (Polak-Ribiere)
        grad_norm_sq = np.sum(grad_t * grad_t)
        if grad_norm_sq > 1e-30 and prev_grad_t is not None:
            dot = np.sum(grad_t * prev_grad_t)
            beta = max(0.0, dot / grad_norm_sq)
            d_new = -grad_t + beta * d
        else:
            d_new = -grad_t.copy()
            beta = 0.0
        
        # Project to tangent space
        radial_d = np.sum(d_new * x, axis=1, keepdims=True)
        d = d_new - radial_d * x
        
        # Store for next BB computation
        prev_grad_t = grad_t.copy()
        prev_step = x.copy()
    
    return best_x, best_e


def _basin_hopping_rpg(x0, n_restarts=8, deadline=None):
    """Basin hopping with Riemannian PG step sizes.
    
    Uses random perturbations followed by Riemannian CG with BB step sizes.
    """
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    rng = np.random.RandomState(123)
    
    # Adaptive perturbation scale
    pert_scale = 0.12
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = rng.randn(N, 3) * pert_scale
        x_trial = _project_to_sphere(x + perturbation)
        
        # Local minimization with Riemannian CG + BB
        x_opt, e_opt = _riemannian_cg_bb(x_trial, max_iter=2000, tol=1e-13, deadline=deadline)
        
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
            # Accept: keep or slightly increase perturbation
            pert_scale = min(pert_scale * 1.15, 0.25)
        else:
            # Reject: decrease perturbation
            pert_scale = max(pert_scale * 0.75, 0.03)
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    x_fib = _fibonacci_sphere(N)
    initializations.append(x_fib)
    
    # 2. Multiple random seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        x_rand = _random_uniform_sphere(N, seed=seed)
        initializations.append(x_rand)
    
    # 3. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    x_fib_pert = _project_to_sphere(x_fib + rng.randn(N, 3) * 0.05)
    initializations.append(x_fib_pert)
    
    best_x = initializations[0].copy()
    best_e = _energy(best_x)
    
    # Optimize each initialization with Riemannian CG + BB
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        try:
            x_opt, e_opt = _riemannian_cg_bb(x_init, max_iter=1500, tol=1e-12, deadline=deadline - 20)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
    
    # Basin hopping with Riemannian CG + BB from best configuration
    if time.time() < deadline - 30:
        try:
            x_bh, e_bh = _basin_hopping_rpg(best_x, n_restarts=10, deadline=deadline - 15)
            if e_bh < best_e:
                best_x = x_bh
                best_e = e_bh
        except Exception:
            pass
    
    # Final tight polish with very tight tolerance
    if time.time() < deadline - 10:
        try:
            x_final, e_final = _riemannian_cg_bb(best_x, max_iter=5000, tol=1e-15, deadline=deadline - 5)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
    
    # Ultra-fine final polish
    if time.time() < deadline - 3:
        try:
            x_ultra, e_ultra = _riemannian_cg_bb(best_x, max_iter=3000, tol=1e-16, deadline=deadline - 0.5)
            if e_ultra < best_e:
                best_x = x_ultra
                best_e = e_ultra
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
