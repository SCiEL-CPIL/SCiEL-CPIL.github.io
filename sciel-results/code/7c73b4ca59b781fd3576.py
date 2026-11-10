"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy_precise(pts):
    """Compute logarithmic energy using math.fsum for precision"""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            terms.append(-math.log(d))
    return math.fsum(terms)

def _log_energy_vec(pts):
    """Vectorized logarithmic energy"""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, 1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    return float(np.sum(-np.log(d)))

def _exp_map(pts, deltas):
    """Exponential map on the sphere: move pts by tangent vectors deltas"""
    # deltas are tangent vectors (orthogonal to pts)
    # For small angles, exp map ≈ pts + deltas
    # For general case: use Rodrigues-like formula
    norms_d = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms_d = np.maximum(norms_d, 1e-15)
    angles = np.sqrt(np.sum(deltas**2, axis=1))
    # Unit tangent direction
    u = deltas / norms_d
    # Rodrigues rotation formula
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)
    # pts_new = cos(a)*pts + sin(a)*u + (1-cos(a))*(pts·u)*pts
    # But u is already tangent, so pts·u = 0
    result = cos_a[:, np.newaxis] * pts + sin_a[:, np.newaxis] * u
    return result

def _log_of_exp_map(pts, deltas):
    """Logarithmic energy as function of tangent space displacements"""
    new_pts = _exp_map(pts, deltas)
    # Ensure unit norm
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return _log_energy_vec(new_pts)

def _log_of_exp_map_grad(pts, deltas):
    """Gradient of log energy w.r.t. tangent displacements"""
    new_pts = _exp_map(pts, deltas)
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    
    # Compute gradient on sphere
    n = new_pts.shape[0]
    diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(grad * new_pts, axis=1)
    grad_tangent = grad - dot[:, np.newaxis] * new_pts
    return grad_tangent

def _optimize_from_init(pts_init, time_budget, rng):
    """Optimize starting from pts_init using L-BFGS-B in tangent space"""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # First do a few steps of projected gradient descent to get close
    n = pts_init.shape[0]
    pts = pts_init.copy()
    
    # Initial gradient descent phase
    for step in range(200):
        if time.time() > deadline - 5:
            break
        grad = _log_of_exp_map_grad(pts, np.zeros((n, 3)))
        # Simple gradient descent
        lr = 0.01
        new_pts = pts - lr * grad
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        pts = new_pts
    
    # Now use L-BFGS-B in tangent space
    best_energy = _log_energy_vec(pts)
    best_pts = pts.copy()
    
    # Flatten for L-BFGS
    x0 = np.zeros(3 * N)
    
    def fun_and_grad(x_flat):
        deltas = x_flat.reshape(N, 3)
        # Project deltas to be tangent to pts
        dots = np.sum(deltas * pts, axis=1)
        deltas_tangent = deltas - dots[:, np.newaxis] * pts
        
        energy = _log_of_exp_map(pts, deltas_tangent)
        grad = _log_of_exp_map_grad(pts, deltas_tangent)
        return energy, grad.flatten()
    
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        if time.time() < deadline - 3:
            deltas_opt = result.x.reshape(N, 3)
            dots = np.sum(deltas_opt * pts, axis=1)
            deltas_opt = deltas_opt - dots[:, np.newaxis] * pts
            new_pts = _exp_map(pts, deltas_opt)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            new_energy = _log_energy_vec(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_energy

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _polish_high_precision(pts, max_iter=5000, tol=1e-14):
    """High-precision polish using very small steps"""
    n = pts.shape[0]
    pts = pts.copy()
    best_energy = _log_energy_vec(pts)
    best_pts = pts.copy()
    lr = 1e-4
    
    for it in range(max_iter):
        grad = _log_of_exp_map_grad(pts, np.zeros((n, 3)))
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = pts - alpha * grad
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            new_energy = _log_energy_vec(new_pts)
            if new_energy < best_energy - 1e-16:
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            pts = new_pts
            best_energy = new_energy
            best_pts = new_pts.copy()
            if best_energy > _log_energy_vec(pts) - 1e-15:
                break
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin for 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    init_configs = []
    
    # Fibonacci spiral
    init_configs.append(_fibonacci_spiral(N))
    
    # Random initializations
    for _ in range(8):
        init_configs.append(_random_points(N, rng))
    
    # Perturbed Fibonacci
    for _ in range(8):
        base = _fibonacci_spiral(N)
        noise_scale = 0.05 + 0.1 * rng.random()
        noise = rng.standard_normal((N, 3)) * noise_scale
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        init_configs.append(perturbed / norms)
    
    # Antipodal-symmetric perturbation
    for _ in range(4):
        base = _fibonacci_spiral(N)
        # Add small random perturbations that preserve some symmetry
        noise = rng.standard_normal((N, 3)) * 0.03
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        init_configs.append(perturbed / norms)
    
    # Allocate time budget
    n_inits = len(init_configs)
    time_per_init = (deadline - start_time - 10.0) / n_inits
    time_per_init = max(time_per_init, 2.0)  # minimum 2s per init
    
    for init_idx, init_pts in enumerate(init_configs):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Time budget for this initialization
        budget = min(time_per_init, remaining - 5.0)
        if budget < 1.0:
            break
        
        result_pts, result_energy = _optimize_from_init(
            init_pts, budget, rng
        )
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts.copy()
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0 and best_pts is not None:
        remaining = deadline - elapsed - 2.0
        if remaining > 2.0:
            final_pts, final_energy = _polish_high_precision(best_pts, max_iter=3000)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
    
    # Verify and clean up
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm and check validity
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for near-duplicate points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
