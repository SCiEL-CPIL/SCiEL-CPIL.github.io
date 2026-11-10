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
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient w.r.t. free R^3 coordinates.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    inv_dist_sq = np.zeros_like(dist_sq)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    # Gradient of E = -sum log(d) is -sum (x_i - x_j)/d^2
    # diff[i,j] = x_i - x_j
    # grad_i = - sum_j diff[i,j] / dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _adam_optimize(pts0, time_limit=100.0, lr_init=0.01, no_improve_break=500):
    """Optimize points on the sphere using Adam gradient descent with adaptive learning rate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Adam parameters
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Initialize moments
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    no_improve_steps = 0
    max_no_improve = no_improve_break
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if no_improve_steps > max_no_improve:
            break
            
        # Project gradient to tangent space
        grad_tangent = _project_to_tangent(pts, grad)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent * grad_tangent)
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        pts_new = pts - step
        
        # Retract to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check for convergence or divergence
        if not np.all(np.isfinite(pts_new)):
            break
            
        pts = pts_new
        
        # Adaptive learning rate: reduce if energy is increasing
        if t % 50 == 0:
            current_energy = _log_energy(pts)
            if current_energy > best_energy + 1.0:
                lr *= 0.95
                if lr < 1e-7:
                    break

    return best_pts, best_energy

def _lbfgs_optimize(pts0, maxiter=2000, ftol=1e-15, gtol=1e-12):
    """Optimize using L-BFGS-B with tight tolerances, starting from pts0."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.fun < best_energy:
            pts_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            pts_opt = pts_opt / norms
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    return best_pts, best_energy

def _lbfgs_polish(pts0, maxiter=500, ftol=1e-14, gtol=1e-10):
    """Polish solution using L-BFGS-B with tight tolerances."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Flatten to 1D for scipy
    def objective(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        return e
    
    def gradient(x_flat):
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        # Project to tangent and flatten
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 3, 'ftol': ftol, 'gtol': gtol}
        )
        
        if result.success or result.fun < best_energy:
            pts_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            pts_opt = pts_opt / norms
            e_opt = _log_energy(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 147.0  # 3 second safety margin
    
    # Start with Fibonacci spiral
    pts0 = _make_fibonacci_sphere()
    
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Phase 1: L-BFGS from Fibonacci (fast convergence)
    if time.time() < deadline - 10.0:
        pts_opt, energy_opt = _lbfgs_optimize(pts0, maxiter=3000, ftol=1e-15, gtol=1e-12)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 2: Multi-restart with L-BFGS from diverse starts
    rng = np.random.RandomState(42)
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate diverse starting points
        if r == 0:
            pts_rand = pts0 + 0.05 * rng.randn(N, 3)
        elif r == 1:
            pts_rand = pts0 + 0.02 * rng.randn(N, 3)
        elif r == 2:
            pts_rand = pts0 + 0.1 * rng.randn(N, 3)
        elif r == 3:
            pts_rand = pts0 + 0.2 * rng.randn(N, 3)
        elif r % 5 == 4:
            pts_rand = _random_sphere(N, rng)
        elif r % 5 == 3:
            # Perturbed best solution
            pts_rand = best_pts + 0.03 * rng.randn(N, 3)
        else:
            # Various perturbation levels around Fibonacci
            scale = 0.01 + 0.05 * (r % 3)
            pts_rand = pts0 + scale * rng.randn(N, 3)
        
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        # Use L-BFGS for each restart (faster than Adam for this problem)
        maxiter = min(3000, max(500, int(remaining * 20)))
        pts_opt, energy_opt = _lbfgs_optimize(pts_rand, maxiter=maxiter, ftol=1e-14, gtol=1e-11)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 3: Adam fine-tuning on best solution
    if time.time() < deadline - 10.0:
        remaining = max(deadline - time.time() - 8.0, 2.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining, lr_init=0.002, no_improve_break=300)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 4: Final L-BFGS polish with very tight tolerances
    if time.time() < deadline - 5.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=5000, ftol=1e-16, gtol=1e-13)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 5: One more ultra-tight L-BFGS pass
    if time.time() < deadline - 2.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=10000, ftol=1e-16, gtol=1e-14)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for very close points and perturb if necessary
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        for _ in range(100):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
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
