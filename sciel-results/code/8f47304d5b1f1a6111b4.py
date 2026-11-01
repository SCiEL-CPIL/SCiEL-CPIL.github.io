import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms_safe
    mask = norms < 1e-15
    sin_norms_over_norms[mask] = 1.0
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _halton_sphere(n, base=2, base2=3):
    """Generate Halton sequence points on sphere."""
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
        u = halton(i, base)
        v = halton(i, base2)
        z = 2.0 * u - 1.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * v
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _log_energy_fast(current), _log_energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Backtracking line search with Armijo condition
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        # Try multiple starting alphas
        for alpha_init in [0.5, 0.1, 0.01]:
            alpha = alpha_init
            for ls in range(20): # Reduced line search steps
                new_pts = _exponential_map(current, alpha * d)
                new_energy = _log_energy_fast(new_pts)
                
                # Armijo condition
                if new_energy < energy - 1e-4 * alpha * np.sum(grad * d):
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = _log_energy_grad(new_pts)
                    break
                
                alpha *= 0.7
            
            if best_alpha > 0:
                break
        
        if best_alpha == 0.0:
            # Try steepest descent
            alpha = 0.01
            for ls in range(10):
                new_pts = _exponential_map(current, alpha * (-grad))
                new_energy = _log_energy_fast(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = _log_energy_grad(new_pts)
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break
        
        # Update
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate a smaller set of high-quality initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere
    halton_pts = _halton_sphere(N)
    initializations.append(halton_pts.copy())
    
    # 3. A few random inits
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
        
    # 4. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.05
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)

    # Sort initializations by initial energy (best first)
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]
    
    # Process initializations with time budget
    for idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 5:
            break
        
        # Allocate time based on rank
        if idx == 0:
            cg_time = min(30, remaining * 0.5)
        elif idx < 3:
            cg_time = min(15, remaining * 0.3)
        else:
            cg_time = min(5, remaining * 0.2)
        
        try:
            # Use Riemannian CG which is faster and more controllable than L-BFGS-B for this size
            current_pts, current_energy = _riemannian_cg(init_pts, time_limit=cg_time)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 2:
        try:
            current_pts, current_energy = _riemannian_cg(best_pts, time_limit=min(5, deadline - time.time()))
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts
        except Exception:
            pass

    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
