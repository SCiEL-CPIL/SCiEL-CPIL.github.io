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
    deadline = time.time() + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    pts0 = _make_fibonacci_sphere()
    
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Main optimization from Fibonacci start
    remaining = deadline - time.time()
    if remaining > 15.0:
        pts_opt, energy_opt = _adam_optimize(pts0, time_limit=remaining - 20.0, lr_init=0.01, no_improve_break=800)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Try multiple diverse restarts with remaining time
    np.random.seed(42)
    rng = np.random.RandomState(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 12.0:
            break
        
        if r == 0:
            # Slightly perturbed Fibonacci
            pts_rand = pts0 + 0.03 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        elif r == 1:
            # Very slightly perturbed Fibonacci
            pts_rand = pts0 + 0.01 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        elif r % 4 == 2:
            # Random points
            pts_rand = _random_sphere(N, rng)
        elif r % 4 == 3:
            # More perturbed Fibonacci
            pts_rand = pts0 + 0.1 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        else:
            # Moderate perturbation
            pts_rand = pts0 + 0.05 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Time allocation: decreasing time for later restarts
        t_limit = max(remaining * 0.55, 2.0)
        pts_opt, energy_opt = _adam_optimize(pts_rand, time_limit=t_limit, lr_init=0.008, no_improve_break=400)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Final fine-tuning with Adam (short, high precision)
    if time.time() < deadline - 8.0:
        remaining = max(deadline - time.time() - 5.0, 1.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining, lr_init=0.003, no_improve_break=200)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # L-BFGS polishing with tight tolerances
    if time.time() < deadline - 3.0:
        pts_opt, energy_opt = _lbfgs_polish(best_pts, maxiter=500, ftol=1e-14, gtol=1e-10)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # One more very tight L-BFGS pass
    if time.time() < deadline - 1.5:
        pts_opt, energy_opt = _lbfgs_polish(best_pts, maxiter=1000, ftol=1e-15, gtol=1e-12)
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
