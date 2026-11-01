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
    # For zero norms, sin(0)/0 = 1
    mask = norms < 1e-15
    sin_norms_over_norms[mask] = 1.0
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_cg(pts, time_limit=15.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _log_energy_fast(current), _log_energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
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
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            # Try smaller step with steepest descent
            alpha = 0.01
            for ls in range(20):
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
        if np.linalg.norm(grad) < 1e-15:
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
        if time.time() - start > time_limit - 0.5:
            raise ValueError("Time limit exceeded")
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy = _log_energy_fast(p)
        grad = _log_energy_grad(p)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13,
                                'maxfun': 15000, 'maxls': 50})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts.copy(), _log_energy_fast(pts)

def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-3
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-17:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.05, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-20:
                break
    
    return current, energy

def _gram_matrix_optimize(pts, time_limit=20.0):
    """
    Optimize using Gram matrix representation.
    The Gram matrix G = P^T P where P is (3, N).
    We optimize on the manifold of rank-3 PSD matrices.
    """
    start = time.time()
    
    # Initial Gram matrix
    G = pts.T @ pts  # (N, N)
    G = (G + G.T) / 2.0
    np.fill_diagonal(G, 1.0)
    
    def _gram_energy_and_grad(G):
        """Compute energy and gradient w.r.t. Gram matrix entries."""
        # Distances squared: d_ij^2 = 2 - G_ij - G_ji = 2 - 2*G_ij for symmetric G
        d2 = 2.0 - 2.0 * G
        np.fill_diagonal(d2, 0.0)
        d2 = np.maximum(d2, 1e-15)
        
        # Energy = sum_{i<j} log(1/sqrt(d2_ij)) = -0.5 * sum_{i<j} log(d2_ij)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        energy = -0.5 * np.sum(np.log(d2_iu))
        
        # Gradient: d(energy)/d(G_ij) = -0.5 * (-1/d2_ij) * (-2) = 1/d2_ij for i<j
        # But we need to be careful with the chain rule
        # E = -0.5 * sum log(d2_ij), d2_ij = 2 - 2*G_ij
        # dE/dG_ij = -0.5 * (1/d2_ij) * (-2) = 1/d2_ij
        grad_G = np.zeros_like(G)
        grad_G[iu] = 1.0 / d2_iu
        grad_G = grad_G + grad_G.T - np.diag(np.diag(grad_G))
        
        # For the diagonal, d2_ii = 0, so no contribution
        np.fill_diagonal(grad_G, 0.0)
        
        return energy, grad_G
    
    # Use L-BFGS-B on the Gram matrix entries (upper triangle)
    def _gram_to_vec(G):
        iu = np.triu_indices(N, k=1)
        return G[iu].copy()
    
    def _vec_to_gram(v):
        G = np.zeros((N, N))
        iu = np.triu_indices(N, k=1)
        G[iu] = v
        G = G + G.T
        np.fill_diagonal(G, 1.0)
        return G
    
    def _gram_obj(v):
        if time.time() - start > time_limit - 0.5:
            raise ValueError("Time limit exceeded")
        G = _vec_to_gram(v)
        # Ensure PSD by clipping eigenvalues
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 0.0)
        G_psd = (eigvecs * eigvals) @ eigvecs.T
        G_psd = (G_psd + G_psd.T) / 2.0
        np.fill_diagonal(G_psd, 1.0)
        
        energy, grad_G = _gram_energy_and_grad(G_psd)
        
        # Project gradient back to the parameterization
        iu = np.triu_indices(N, k=1)
        grad_v = grad_G[iu]
        
        return energy, grad_v
    
    v0 = _gram_to_vec(G)
    
    try:
        res = minimize(_gram_obj, v0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 5000, 'maxls': 30})
        
        v_opt = res.x
        G_opt = _vec_to_gram(v_opt)
        
        # Recover points from Gram matrix via eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G_opt)
        # Keep only top 3 eigenvalues
        idx = np.argsort(eigvals)[::-1][:3]
        eigvals_top = eigvals[idx]
        eigvecs_top = eigvecs[:, idx]
        
        # P = eigvecs_top * sqrt(eigvals_top)
        sqrt_eigvals = np.sqrt(np.maximum(eigvals_top, 0.0))
        P = eigvecs_top * sqrt_eigvals  # (N, 3)
        
        pts_opt = _normalize(P)
        energy = _log_energy_fast(pts_opt)
        
        return pts_opt, energy
    except Exception:
        return pts.copy(), _log_energy_fast(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere
    halton_pts = _halton_sphere(N)
    initializations.append(halton_pts.copy())
    
    # 3. Random uniform on sphere
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Perturbed Halton
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = halton_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 6. Slightly different Fibonacci variants
    for offset in [0.3, 0.7, 0.15, 0.85]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    # 7. Spherical spiral
    i = np.arange(N)
    z = 1.0 - 2.0 * i / (N - 1)
    phi = np.linspace(0, 100 * np.pi, N)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(_normalize(pts))
    
    # Sort initializations by initial energy (best first)
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]
    init_energies = [init_energies[i] for i in order]
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        # Phase 1: L-BFGS-B for coarse optimization
        try:
            current_pts, current_energy = _lbfgs_optimize(init_pts, time_limit=15.0)
            
            # Phase 2: Riemannian CG for fine optimization
            if time.time() < deadline - 15:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=8.0)
                
                # Phase 2b: Another L-BFGS-B after CG
                if time.time() < deadline - 10:
                    current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=5.0)
            
            # Phase 3: Gram matrix optimization
            if time.time() < deadline - 12:
                gram_pts, gram_energy = _gram_matrix_optimize(current_pts, time_limit=8.0)
                if gram_energy < current_energy:
                    current_pts = gram_pts
                    current_energy = gram_energy
                    
                    # Polish after Gram optimization
                    if time.time() < deadline - 8:
                        current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=4.0)
                        if time.time() < deadline - 5:
                            current_pts, current_energy = _riemannian_cg(current_pts, time_limit=3.0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Tight L-BFGS-B polish
        try:
            x0 = current_pts.flatten()
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 8000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 80
                }
            )
            polished_pts = _normalize(res.x.reshape(N, 3))
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                current_energy = polished_energy
                current_pts = polished_pts
        except Exception:
            pass
        
        # Riemannian CG polish
        if time.time() < deadline - 3:
            current_pts, current_energy = _riemannian_cg(current_pts, time_limit=2.0)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts
    
    # Final tiny-step polish
    if time.time() < deadline - 1:
        best_pts, best_energy = _polish(best_pts, time_limit=2.0)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
