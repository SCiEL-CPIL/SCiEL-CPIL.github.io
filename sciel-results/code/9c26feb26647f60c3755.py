"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient with respect to x_i is projected onto the tangent space.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(n, dtype=bool)
    
    # Energy
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_j - x_i) / d_ij^2
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    term1 = inv_dist_sq @ pts  # (N, 3)
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    term2 = pts * row_sums[:, np.newaxis]
    
    grad = term1 - term2  # (N, 3)
    
    # Project to tangent space
    dot_products = np.sum(grad * pts, axis=1)
    grad_tangent = grad - dot_products[:, np.newaxis] * pts
    
    return energy, grad_tangent


def _energy_fsum(pts):
    """Compute energy using math.fsum for high precision."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = ~np.eye(n, dtype=bool)
    dist = np.sqrt(dist_sq[mask])
    log_dists = np.log(dist)
    return -math.fsum(log_dists.tolist())


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_init():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _run_lbfgs_polish(pts, deadline, max_iter=500, tol=1e-12):
    """Polish using L-BFGS-B with penalty for sphere constraint."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        # Penalty for sphere constraint
        norms = np.linalg.norm(p, axis=1)
        penalty = 1e4 * np.sum((norms - 1.0) ** 2)
        # Energy
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        mask = ~np.eye(N, dtype=bool)
        log_dists = np.log(dist)
        log_dists[~mask] = 0.0
        energy = -0.5 * np.sum(log_dists)
        return energy + penalty
    
    def gradient(x):
        p = x.reshape(N, 3)
        # Energy gradient
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        
        term1 = inv_dist_sq @ p
        row_sums = np.sum(inv_dist_sq, axis=1)
        term2 = p * row_sums[:, np.newaxis]
        energy_grad = (term1 - term2).flatten()
        
        # Penalty gradient
        norms = np.linalg.norm(p, axis=1)
        penalty_grad = 2e4 * (norms - 1.0) * p / np.maximum(norms, 1e-15)
        
        return energy_grad + penalty_grad.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    
    if result.success or result.fun < objective(x0):
        pts_opt = result.x.reshape(N, 3)
        pts_opt = _project_to_sphere(pts_opt)
        return pts_opt
    return pts


def _run_riemannian_gd(pts, deadline, max_iters=2000):
    """Riemannian gradient descent with momentum and adaptive step size."""
    pts = _project_to_sphere(pts.copy())
    current_energy = _energy_fsum(pts)
    
    lr = 0.005
    momentum = 0.0
    vel = np.zeros_like(pts)
    best_energy = current_energy
    best_pts = pts.copy()
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Adaptive learning rate based on gradient norm
        adaptive_lr = lr * 0.1 / (grad_norm + 1e-8)
        
        vel = momentum * vel - adaptive_lr * grad
        new_pts = pts + vel
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy_fsum(new_pts)
        
        if new_energy < current_energy - 1e-15:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            momentum = 0.9 if it > 50 else 0.8
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        else:
            no_improve_count += 1
            lr *= 0.7
            momentum = 0.0
            vel = np.zeros_like(pts)
        
        if no_improve_count > 50:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial: Fibonacci spiral
    pts = _fibonacci_init()
    best_pts = pts.copy()
    best_energy = _energy_fsum(pts)
    
    # Phase 1: Riemannian gradient descent
    pts_gd, energy_gd = _run_riemannian_gd(pts, deadline, max_iters=1500)
    if energy_gd < best_energy:
        best_pts = pts_gd
        best_energy = energy_gd
    
    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 15.0:
        pts_lbfgs = _run_lbfgs_polish(best_pts, deadline, max_iter=300, tol=1e-14)
        energy_lbfgs = _energy_fsum(pts_lbfgs)
        if energy_lbfgs < best_energy:
            best_pts = pts_lbfgs
            best_energy = energy_lbfgs
    
    # Phase 3: Another round of GD + LBFGS from a random restart if time allows
    rng = np.random.RandomState(42)
    if time.time() < deadline - 30.0:
        random_pts = rng.randn(N, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        pts_gd2, energy_gd2 = _run_riemannian_gd(random_pts, deadline, max_iters=800)
        if energy_gd2 < best_energy:
            best_pts = pts_gd2
            best_energy = energy_gd2
        
        if time.time() < deadline - 15.0:
            pts_lbfgs2 = _run_lbfgs_polish(best_pts, deadline, max_iter=200, tol=1e-14)
            energy_lbfgs2 = _energy_fsum(pts_lbfgs2)
            if energy_lbfgs2 < best_energy:
                best_pts = pts_lbfgs2
                best_energy = energy_lbfgs2
    
    # Final polish if time allows
    if time.time() < deadline - 10.0:
        pts_final = _run_lbfgs_polish(best_pts, deadline, max_iter=100, tol=1e-15)
        energy_final = _energy_fsum(pts_final)
        if energy_final < best_energy:
            best_pts = pts_final
            best_energy = energy_final
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_init()
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_init()
    
    return best_pts
# EVOLVE-BLOCK-END
