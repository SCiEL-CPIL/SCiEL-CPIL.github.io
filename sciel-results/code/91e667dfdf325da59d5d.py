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

def _log_energy_knn(pts, k):
    """Compute log energy using only k-nearest neighbors for coarse optimization."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each point, find k nearest neighbors
    total_energy = 0.0
    for i in range(n):
        sorted_dists = np.sort(dists[i, :])
        # Take k nearest (excluding self which is inf)
        knn_dists = sorted_dists[:k]
        knn_dists = np.maximum(knn_dists, 1e-15)
        total_energy += np.sum(np.log(1.0 / knn_dists))
    return total_energy / 2.0  # Each pair counted twice

def _log_energy_knn_grad(pts, k):
    """Compute gradient of k-NN log energy (approximation for coarse optimization)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    grad = np.zeros_like(pts)
    
    for i in range(n):
        # Find k nearest neighbors
        sorted_idx = np.argsort(dists_sq[i, :])
        knn_idx = sorted_idx[:k]
        
        for j in knn_idx:
            d2 = dists_sq[i, j]
            if d2 < 1e-30:
                continue
            # Contribution to grad[i] from pair (i,j)
            # d/d(pts[i]) log(1/|pts[i]-pts[j]|) = -(pts[i]-pts[j])/|pts[i]-pts[j]|^2
            grad[i] += -(pts[i] - pts[j]) / d2
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _energy_and_grad_3d(x):
    """Energy and gradient for L-BFGS-B on 3D coordinates."""
    pts = x.reshape(N, 3)
    # Normalize points to ensure they lie on the unit sphere
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

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _log_energy_fast(current), _log_energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = None
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search with Armijo condition
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        # Compute directional derivative
        dir_deriv = np.sum(grad * d)
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            new_grad = _log_energy_grad(new_pts)
            
            # Armijo condition: f(new) <= f(old) + c * alpha * grad^T * d
            c1 = 1e-4
            armijo_ok = new_energy <= energy + c1 * alpha * dir_deriv
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if armijo_ok:
                break
            
            alpha *= 0.5
        
        if best_alpha == 0.0:
            # Try a steepest descent step
            alpha = 0.01
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * (-grad))
                new_energy = _log_energy_fast(new_pts)
                new_grad = _log_energy_grad(new_pts)
                if new_energy < energy:
                    best_pts = new_pts
                    best_energy = new_energy
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                alpha *= 0.5
            
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
        if prev_grad is not None:
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
    """High-precision polish with Newton-type steps and very small gradient steps."""
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
        
        # Try Newton step first: approximate Hessian as diag of gradient components
        # For sphere constraint, use projected gradient method with adaptive step
        alpha = step_size
        accepted = False
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-15:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size if we made good progress
            if new_energy < energy - 1e-10:
                step_size = min(alpha * 1.5, 0.1)
            else:
                step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
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
        if time.time() > deadline - 25:
            break
        
        try:
            current_pts = init_pts.copy()
            
            # Phase 0: Coarse optimization with k-NN approximation (multigrid-like)
            # First optimize with fewer neighbors to resolve long-range structure
            k_coarse = 10
            current_pts_c = current_pts.copy()
            start_c = time.time()
            while time.time() - start_c < 5.0:
                grad_c = _log_energy_knn_grad(current_pts_c, k_coarse)
                if np.linalg.norm(grad_c) < 1e-12:
                    break
                # Gradient descent with backtracking
                alpha = 0.01
                energy_c = _log_energy_knn(current_pts_c, k_coarse)
                for _ in range(20):
                    new_pts = current_pts_c - alpha * grad_c
                    new_pts = _normalize(new_pts)
                    new_energy_c = _log_energy_knn(new_pts, k_coarse)
                    if new_energy_c < energy_c:
                        current_pts_c = new_pts
                        energy_c = new_energy_c
                        break
                    alpha *= 0.5
                else:
                    break
            
            # Use coarse result as starting point for full optimization
            current_pts = current_pts_c
            
            # Phase 1: L-BFGS-B for coarse optimization
            current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=15.0)
            
            # Phase 2: Riemannian CG for fine optimization
            if time.time() < deadline - 15:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=10.0)
            
            # Phase 3: Final L-BFGS-B polish
            if time.time() < deadline - 8:
                current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=5.0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish with smaller steps
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Use a very tight L-BFGS-B polish
        try:
            x0 = current_pts.flatten()
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 100
                }
            )
            polished_pts = _normalize(res.x.reshape(N, 3))
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
    
    # Final tiny-step polish
    if time.time() < deadline - 2:
        best_pts, best_energy = _polish(best_pts, time_limit=3.0)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
