"""Combined approach: L-BFGS-B + Riemannian CG polish for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected to tangent space)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = -sum_j (pts[i,k] - pts[j,k]) / dists_sq[i,j]
    # = -pts[i,k] * sum_j(1/d2) + sum_j(pts[j,k] / d2)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad


def _energy_and_grad_3d(x):
    """Energy and gradient for L-BFGS-B on 3D coordinates."""
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy = _log_energy_fast(pts)
    grad = _log_energy_grad(pts)
    return energy, grad.flatten()


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    grad = _log_energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = _log_energy_grad(new_pts)
            
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        # Update
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=25.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy = _log_energy_fast(p)
        grad = _log_energy_grad(p)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 15000, 'maxls': 50})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=8.0):
    """High-precision polish with very small steps using Riemannian CG."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Use Riemannian CG for high-precision polish
    current, energy = _riemannian_cg(current, time_limit=time_limit, max_iter=500)
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Random uniform on sphere
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 4. Slightly different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: L-BFGS-B for coarse optimization
        try:
            time_left_for_lbfgs = min(deadline - time.time() - 15, 25.0)
            if time_left_for_lbfgs < 5:
                break
            current_pts, current_energy = _lbfgs_optimize(init_pts, time_limit=time_left_for_lbfgs)
            
            # Phase 2: Riemannian CG for fine optimization
            time_left_for_cg = min(deadline - time.time() - 8, 12.0)
            if time_left_for_cg > 2:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=time_left_for_cg)
            
            # Phase 3: Final polish
            time_left_for_polish = min(deadline - time.time() - 3, 6.0)
            if time_left_for_polish > 1:
                current_pts, current_energy = _polish(current_pts, time_limit=time_left_for_polish)
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish on best solution
    if best_pts is not None and time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 1, 5.0)
        if final_time > 0.5:
            try:
                final_pts, final_energy = _riemannian_cg(best_pts, time_limit=final_time, max_iter=200)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
            except Exception:
                pass
    
    # Final normalization
    if best_pts is not None:
        best_pts = _normalize(best_pts)
    else:
        best_pts = _normalize(fib_pts)
    
    return best_pts
