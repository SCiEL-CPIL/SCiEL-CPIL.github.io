"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
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
    pts = _normalize(pts)
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
    return _normalize(pts)

def _halton_sphere(n):
    """Generate Halton sequence points projected onto the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        u1 = halton(i + 1, 2)
        u2 = halton(i + 1, 3)
        u3 = halton(i + 1, 5)
        pts[i] = [u1 - 0.5, u2 - 0.5, u3 - 0.5]
    
    return _normalize(pts)

def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    vecs = rng.normal(size=(n, 3))
    return _normalize(vecs)

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
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
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            new_grad = _log_energy_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
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

def _lbfgs_optimize(pts, time_limit=30.0):
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
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 10000, 'maxls': 50})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy

def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-4
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy

def _spectral_multigrid(pts, time_limit=20.0):
    """
    Coarse-to-fine optimization using spectral gap-based repulsion.
    This approximates long-range interactions first by using a k-NN graph
    and its Laplacian, then refines with full energy.
    """
    start = time.time()
    current = pts.copy()
    
    # Phase 1: Coarse optimization using k-NN graph Laplacian
    # Build k-NN graph
    k = 10
    from scipy.spatial.distance import cdist
    
    dists = cdist(current, current)
    np.fill_diagonal(dists, np.inf)
    
    # Get k nearest neighbors
    knn_indices = np.argsort(dists, axis=1)[:, :k]
    
    # Build Laplacian
    n = len(current)
    L = np.zeros((n, n))
    for i in range(n):
        for j_idx in knn_indices[i]:
            d = dists[i, j_idx]
            w = 1.0 / (d * d)  # Log energy weight
            L[i, i] += w
            L[i, j_idx] -= w
    
    # Symmetrize
    L = (L + L.T) / 2.0
    
    # Compute pseudo-inverse of Laplacian for coarse force
    # Add small regularization
    L_reg = L + 1e-10 * np.eye(n)
    
    # Get gradient of full energy
    grad = _log_energy_grad(current)
    
    # Apply coarse force: use L^{-1} grad as a smoothed direction
    try:
        coarse_force = np.linalg.solve(L_reg, grad)
    except:
        coarse_force = grad
    
    # Project to tangent space
    dot_prod = np.sum(coarse_force * current, axis=1)
    coarse_force -= dot_prod[:, np.newaxis] * current
    
    # Take a few steps in the coarse direction
    energy = _log_energy_fast(current)
    alpha = 0.01
    for _ in range(10):
        if time.time() - start > time_limit - 5.0:
            break
        new_pts = current - alpha * coarse_force
        new_pts = _normalize(new_pts)
        new_energy = _log_energy_fast(new_pts)
        if new_energy < energy:
            current = new_pts
            energy = new_energy
            alpha *= 0.9
        else:
            alpha *= 0.5
    
    # Phase 2: Fine optimization with full gradient
    if time.time() - start < time_limit - 2.0:
        current, energy = _riemannian_cg(current, time_limit=time_limit - time.time() - 1.0, max_iter=50)
    
    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere
    hal_pts = _halton_sphere(N)
    initializations.append(hal_pts.copy())
    
    # 3. Random Sphere (different seeds)
    for seed in [42, 123, 456]:
        rand_pts = _random_sphere(N, seed=seed)
        initializations.append(rand_pts.copy())
    
    # Optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # Use spectral multigrid for the first few, then standard methods
        if idx < 2 and time.time() < deadline - 30.0:
            # Try spectral multigrid
            try:
                opts_pts, opts_energy = _spectral_multigrid(init_pts, time_limit=20.0)
                if opts_energy < best_energy:
                    best_energy = opts_energy
                    best_pts = opts_pts.copy()
            except Exception:
                pass
        
        # Always try L-BFGS as a robust fallback
        if time.time() < deadline - 15.0:
            try:
                lbfgs_pts, lbfgs_energy = _lbfgs_optimize(init_pts, time_limit=25.0)
                if lbfgs_energy < best_energy:
                    best_energy = lbfgs_energy
                    best_pts = lbfgs_pts.copy()
            except Exception:
                pass
        
        # Polish the best result found so far
        if best_pts is not None and time.time() < deadline - 5.0:
            try:
                polished_pts, polished_energy = _polish(best_pts, time_limit=5.0)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts.copy()
            except Exception:
                pass
    
    # Final polish if we have a result
    if best_pts is not None and time.time() < deadline - 1.0:
        try:
            final_pts, final_energy = _polish(best_pts, time_limit=3.0)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, return Fibonacci points
    if best_pts is None:
        best_pts = fib_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
