"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
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
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
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
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        u3 = halton(i, base_start + 2)
        
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    # For unit sphere, exponential map: exp_p(v) = cos(|v|)p + sin(|v|)/|v| v
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.5  # Start with larger step
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.7  # More aggressive backtracking
        
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
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 10000, 'maxls': 50})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _dual_optimize(pts, time_limit=40.0):
    """Alternate between L-BFGS-B and Riemannian CG."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Phase 1: L-BFGS-B for coarse optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left * 0.6, 25.0))
    
    # Phase 2: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.8, 15.0))
    
    # Phase 3: Final L-BFGS-B polish
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left, 10.0))
    
    return current, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using L-BFGS with tight tolerances."""
    start = time.time()
    current = pts.copy()
    
    # Use L-BFGS-B with very tight tolerances for final polish
    x0 = current.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14,
                                'maxfun': 2000, 'maxls': 30})
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        return x_opt, energy
    except Exception:
        return current, _log_energy_fast(current)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())
    
    # 2. Halton sequence
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())
    
    # 3. Random initializations with different seeds
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci with different scales
    for seed, scale in [(0, 0.02), (1, 0.05), (2, 0.1)]:
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * scale
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Perturbed Halton
    for seed, scale in [(0, 0.03), (1, 0.08)]:
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = pts_halton + rng_local.standard_normal((N, 3)) * scale
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 6. Different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    # Optimize each initialization with dual-representation approach
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        time_left = deadline - time.time() - 5
        if time_left < 8:
            break
        
        # Use dual optimization with adaptive time allocation
        try:
            # Allocate more time to earlier initializations
            time_per_init = min(time_left, 30.0)
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=time_per_init)
            
            # Final polish
            if time.time() < deadline - 3:
                polish_time = min(deadline - time.time() - 1, 4.0)
                if polish_time > 0.5:
                    polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                    if polished_energy < opt_energy:
                        opt_pts = polished_pts
                        opt_energy = polished_energy
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            # Fallback to simple L-BFGS-B
            try:
                time_left = deadline - time.time() - 5
                if time_left > 5:
                    opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 15.0))
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 3:
        final_time = min(deadline - time.time() - 1, 10.0)
        if final_time > 0.5:
            try:
                final_pts, final_energy = _polish(best_pts, time_limit=final_time)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
