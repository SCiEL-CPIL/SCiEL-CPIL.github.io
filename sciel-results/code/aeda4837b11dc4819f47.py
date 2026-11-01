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
    d_sq = np.maximum(d_sq, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(d_sq))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # E = sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij^2)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/d_ij^2) * 2(x_i - x_j)
    #          = -sum_j (x_i - x_j)/d_ij^2
    #          = -x_i * sum_j(1/d_ij^2) + sum_j(x_j/d_ij^2)
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
    small = norms < 1e-8
    norms = np.where(small, 1.0, norms)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    result = cos_norms * pts + sin_norms_over_norms * tangent_vecs
    # For very small steps, use first-order approximation
    result = np.where(small[:, 0, np.newaxis], pts + tangent_vecs, result)
    return _normalize(result)

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

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
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    return pts

def _farthest_point_init(n, seed=42):
    """Generate points using farthest point sampling on the sphere."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance from each candidate to all existing points
        # Use a grid of candidates
        n_candidates = 500
        candidates = rng.standard_normal((n_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Distance from each candidate to closest existing point
        for j in range(i):
            d = np.linalg.norm(candidates - pts[j], axis=1)
            min_dists_cand = np.minimum(min_dists_cand, d) if i > 1 else d.copy()
        
        if i == 1:
            min_dists_cand = np.linalg.norm(candidates - pts[0], axis=1)
        else:
            for j in range(1, i):
                d = np.linalg.norm(candidates - pts[j], axis=1)
                min_dists_cand = np.minimum(min_dists_cand, d)
        
        # Pick the farthest
        idx = np.argmax(min_dists_cand)
        pts[i] = candidates[idx]
    
    return pts

def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
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
        
        # Backtracking line search with strong Wolfe-like condition
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
            
            # Check if we've gone too far (energy increased significantly)
            if new_energy > energy + 0.1 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.85
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha = 0.01
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy = _log_energy_fast(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = _log_energy_grad(new_pts)
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
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy

def _lbfgs_optimize(pts, time_limit=30.0, maxiter=5000, gtol=1e-12):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p_pts = x.reshape(N, 3)
        p_pts = _normalize(p_pts)
        energy = _log_energy_fast(p_pts)
        grad = _log_energy_grad(p_pts)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': gtol,
                            'maxfun': maxiter * 5, 'maxls': 50,
                            'maxcor': 20})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy

def _polish(pts, time_limit=5.0):
    """High-precision gradient polish with adaptive steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-3
    
    for it in range(20000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        best_new_energy = energy
        best_new_pts = current.copy()
        
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_new_energy:
                best_new_energy = new_energy
                best_new_pts = new_pts
                accepted = True
            
            if new_energy > energy:
                break
            
            alpha *= 0.7
        
        if accepted:
            current = best_new_pts
            energy = best_new_energy
            step_size = min(alpha * 1.05, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy

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
    hal_pts = _halton_sphere(N)
    initializations.append(hal_pts)
    
    # 3. Random uniform on sphere (multiple seeds)
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci (multiple perturbations)
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        scale = 0.02 + 0.01 * seed
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * scale
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Farthest point sampling
    for seed in [42, 123, 456]:
        try:
            fp_pts = _farthest_point_init(N, seed=seed)
            initializations.append(fp_pts)
        except Exception:
            pass
    
    # 6. Slightly different Fibonacci variants
    for offset in [0.3, 0.7, 0.1]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    # Sort initializations by initial energy (best first)
    init_energies = []
    for pts in initializations:
        e = _log_energy_fast(pts)
        init_energies.append((e, pts))
    init_energies.sort(key=lambda x: x[0])
    
    for idx, (init_energy, init_pts) in enumerate(init_energies):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # If we've already found a very good solution, be more selective
        if remaining < 15 and best_energy < init_energy:
            break
        
        current_pts = init_pts.copy()
        
        # Phase 1: L-BFGS-B for coarse optimization
        if remaining > 25:
            try:
                t1 = min(20.0, remaining - 15.0)
                current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=t1, maxiter=3000)
            except Exception:
                continue
            
            # Phase 2: Riemannian CG for fine optimization
            if time.time() < deadline - 10:
                try:
                    t2 = min(10.0, deadline - time.time() - 5.0)
                    current_pts, current_energy = _riemannian_cg(current_pts, time_limit=t2)
                except Exception:
                    pass
            
            # Phase 3: Final L-BFGS-B polish
            if time.time() < deadline - 5:
                try:
                    t3 = min(5.0, deadline - time.time() - 2.0)
                    current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=t3, maxiter=2000, gtol=1e-14)
                except Exception:
                    pass
            
            # Phase 4: Gradient polish
            if time.time() < deadline - 1:
                try:
                    t4 = min(2.0, deadline - time.time() - 0.5)
                    current_pts, current_energy = _polish(current_pts, time_limit=t4)
                except Exception:
                    pass
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        # Early termination if we're at a very good energy
        if best_energy < -2364.63:
            break
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Use a very tight L-BFGS-B polish
        try:
            t_final = min(8.0, remaining - 3.0)
            x0 = current_pts.flatten()
            res = minimize(
                lambda x: (_log_energy_fast(_normalize(x.reshape(N,3))), 
                          _log_energy_grad(_normalize(x.reshape(N,3))).flatten()),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 10000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 100,
                    'maxcor': 30
                }
            )
            polished_pts = _normalize(res.x.reshape(N, 3))
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
        
        # Final gradient polish
        remaining = deadline - time.time()
        if remaining > 2:
            try:
                t_polish = min(3.0, remaining - 0.5)
                polished_pts, polished_energy = _polish(best_pts, time_limit=t_polish)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts
            except Exception:
                pass

    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
