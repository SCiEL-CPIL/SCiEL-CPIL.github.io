"""Stronger algorithm for 100-point logarithmic energy on the unit sphere.
Combines Riemannian gradient flow (direct manifold optimization) with L-BFGS-B
polish, targeted perturbations, simulated annealing, and a final high-precision pass.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 145.0  # Hard wall-clock deadline with 5s safety margin


def _time_left():
    return max(0.0, DEADLINE - time.time())


def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision.
    This is the exact scoring function."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Vectorized energy (np.sum, fast) and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy (fast, for comparison during optimization)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    return energy, grad


def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7,
                     lr_decay=0.999, patience=2000):
    """Riemannian gradient descent with backtracking line search and adaptive learning rate.
    Operates directly on the sphere manifold.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            # Riemannian step: move in direction -grad, then re-project
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-10:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points


def _polish(points, time_budget):
    """High-precision polish using L-BFGS-B with tight tolerances.
    Uses the ambient-space formulation with projection."""
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _optimize_single(points, time_budget, maxiter=3000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a single starting point."""
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _energy_incremental(points, i, old_pi, new_pi):
    """
    Compute change in energy when point i moves from old_pi to new_pi.
    Returns delta_energy = E_new - E_old for the pairs involving i.
    """
    old_pi = np.asarray(old_pi)
    new_pi = np.asarray(new_pi)
    other_points = np.delete(points, i, axis=0)
    
    old_dists = np.linalg.norm(other_points - old_pi, axis=1)
    new_dists = np.linalg.norm(other_points - new_pi, axis=1)
    
    old_dists = np.maximum(old_dists, 1e-15)
    new_dists = np.maximum(new_dists, 1e-15)
    
    # Sum log(1/d) for each
    old_contrib = math.fsum(np.log(1.0 / old_dists).tolist())
    new_contrib = math.fsum(np.log(1.0 / new_dists).tolist())
    
    return new_contrib - old_contrib


def _simulated_annealing_refine(points, time_budget):
    """Refine points using fast simulated annealing with incremental energy updates."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy_fsum(best_points)
    T = 0.03
    
    # Precompute some helpers
    eye3 = np.eye(3)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        move_type = np.random.random()
        
        if move_type < 0.7:
            # Single point rotation move
            i = np.random.randint(N)
            old_pi = best_points[i].copy()
            
            angle = np.random.uniform(-1, 1) * T
            axis = np.random.randn(3)
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis /= norm
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * eye3 + s * K + (1 - c) * k_outer_k
            
            new_pi = R @ old_pi
            new_pi = new_pi / np.linalg.norm(new_pi)
            
            # Incremental energy update
            delta_e = _energy_incremental(best_points, i, old_pi, new_pi)
            
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
                best_points[i] = new_pi
                best_energy += delta_e
        else:
            # Pair swap move
            i, j = np.random.choice(N, size=2, replace=False)
            old_pi = best_points[i].copy()
            old_pj = best_points[j].copy()
            
            angle = np.random.uniform(-1, 1) * T * 0.5
            axis = np.random.randn(3)
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis /= norm
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * eye3 + s * K + (1 - c) * k_outer_k
            
            new_pi = R @ old_pj
            new_pj = R @ old_pi
            new_pi = new_pi / np.linalg.norm(new_pi)
            new_pj = new_pj / np.linalg.norm(new_pj)
            
            # Incremental: change in pairs involving i and j
            other_indices = np.array([k for k in range(N) if k != i and k != j])
            other_points = best_points[other_indices]
            
            old_ij_dist = np.linalg.norm(old_pi - old_pj)
            new_ij_dist = np.linalg.norm(new_pi - new_pj)
            
            old_i_dists = np.linalg.norm(other_points - old_pi, axis=1)
            new_i_dists = np.linalg.norm(other_points - new_pi, axis=1)
            old_j_dists = np.linalg.norm(other_points - old_pj, axis=1)
            new_j_dists = np.linalg.norm(other_points - new_pj, axis=1)
            
            old_ij_dists = np.maximum(old_ij_dist, 1e-15)
            new_ij_dists = np.maximum(new_ij_dist, 1e-15)
            old_i_dists = np.maximum(old_i_dists, 1e-15)
            new_i_dists = np.maximum(new_i_dists, 1e-15)
            old_j_dists = np.maximum(old_j_dists, 1e-15)
            new_j_dists = np.maximum(new_j_dists, 1e-15)
            
            delta_e = (math.log(1.0/new_ij_dists) - math.log(1.0/old_ij_dists) +
                      math.fsum(np.log(1.0/new_i_dists).tolist()) - math.fsum(np.log(1.0/old_i_dists).tolist()) +
                      math.fsum(np.log(1.0/new_j_dists).tolist()) - math.fsum(np.log(1.0/old_j_dists).tolist()))
            
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
                best_points[i] = new_pi
                best_points[j] = new_pj
                best_energy += delta_e
        
        T *= 0.9998
        if T < 1e-5:
            T = 0.015
    
    return best_points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project(points)


def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    def _time_left():
        return max(0.0, DEADLINE - time.time())
    
    # Phase 1: Riemannian flow from Fibonacci spiral
    points_fib = _fibonacci_spiral(N)
    best_points = points_fib.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    
    if _time_left() > 5.0:
        budget1 = min(_time_left() * 0.20, 15.0)
        refined = _riemannian_flow(points_fib, budget1)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: Targeted perturbations + L-BFGS
    np.random.seed(42)
    restart_count = 0
    restart_start = time.time()
    restart_time_limit = _time_left() * 0.30
    
    while True:
        elapsed = time.time() - restart_start
        if elapsed > restart_time_limit:
            break
        if _time_left() < 10.0:
            break
        restart_count += 1
        
        # Perturb the best solution slightly rather than pure random
        perturb_scale = 0.05 / (1 + restart_count * 0.1)
        perturbed = best_points + np.random.randn(N, 3) * perturb_scale
        perturbed = _project(perturbed)
        
        budget_per_restart = min(_time_left() * 0.02, 3.0)
        if budget_per_restart < 0.5:
            break
            
        optimized = _optimize_single(perturbed, budget_per_restart, maxiter=2000, ftol=1e-13, gtol=1e-10)
        e, _ = _energy_grad_vec(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Fine-tune the best solution with L-BFGS
    if _time_left() > 5.0:
        remaining = _time_left() * 0.15
        if remaining > 2.0:
            refined = _optimize_single(best_points, remaining, maxiter=6000, ftol=1e-15, gtol=1e-12)
            e, _ = _energy_grad_vec(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 4: Fast simulated annealing with incremental updates
    if _time_left() > 5.0:
        remaining = _time_left() * 0.25
        if remaining > 3.0:
            np.random.seed(789)
            sa_points = _simulated_annealing_refine(best_points, remaining)
            e, _ = _energy_grad_vec(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: Final high-precision L-BFGS polish
    if _time_left() > 5.0:
        remaining = _time_left() * 0.35
        if remaining > 2.0:
            final_opt = _polish(best_points, remaining)
            e, _ = _energy_grad_vec(final_opt)
            if e < best_energy:
                best_energy = e
                best_points = final_opt
    
    # Final projection and validation
    best_points = _project(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy_fsum(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
