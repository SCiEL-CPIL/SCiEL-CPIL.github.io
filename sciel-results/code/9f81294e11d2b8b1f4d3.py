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
        if time.time() - start > time_limit - 0.1:
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
            for ls in range(50):
                new_pts = _exponential_map(current, alpha * d)
                new_energy = _log_energy_fast(new_pts)
                new_grad = _log_energy_grad(new_pts)
                
                # Armijo condition
                if new_energy < energy - 1e-4 * alpha * np.sum(grad * d):
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                
                if new_energy > best_energy:
                    alpha *= 0.7
                else:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    alpha *= 0.7
            
            if best_alpha > 0:
                break
        
        if best_alpha == 0.0:
            # Try steepest descent
            alpha = 0.01
            for ls in range(30):
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
        if grad_norm < 1e-14:
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
                       options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14,
                                'maxfun': 20000, 'maxls': 80})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts.copy(), _log_energy_fast(pts)

def _polish(pts, time_limit=5.0):
    """High-precision polish with adaptive step sizes."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-3
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-18:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size slightly if we made progress
            step_size = min(alpha * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-22:
                break
    
    return current, energy

def _perturb_escape(pts, energy, rng, time_limit=3.0):
    """Try to escape local minimum via perturbation + re-optimization."""
    start = time.time()
    best_pts = pts.copy()
    best_energy = energy
    
    for attempt in range(10):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Random perturbation
        noise = rng.standard_normal((N, 3)) * 0.05
        perturbed = _normalize(pts + noise)
        
        # Quick optimization
        p, e = _riemannian_cg(perturbed, time_limit=1.0, max_iter=100)
        
        if e < best_energy:
            best_pts = p
            best_energy = e
    
    return best_pts, best_energy

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
    
    # 2. Halton Sphere
    halton_pts = _halton_sphere(N)
    initializations.append(halton_pts.copy())
    
    # 3. Random uniform on sphere
    for seed in range(8):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Perturbed Halton
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = halton_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 6. Slightly different Fibonacci variants
    for offset in [0.3, 0.7, 0.15, 0.85, 0.4, 0.6]:
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
    
    # 8. More random with different seeds
    for seed in range(4, 12):
        rng_local = np.random.default_rng(seed * 999 + 42)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # Sort initializations by initial energy (best first)
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]
    init_energies = [init_energies[i] for i in order]
    
    # Process initializations with time budget
    for idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10:
            break
        
        # Allocate time based on rank
        if idx == 0:
            lbfgs_time = min(25, remaining * 0.4)
            cg_time = min(15, remaining * 0.25)
            polish_time = min(5, remaining * 0.1)
        elif idx < 5:
            lbfgs_time = min(15, remaining * 0.3)
            cg_time = min(10, remaining * 0.2)
            polish_time = min(3, remaining * 0.08)
        else:
            lbfgs_time = min(8, remaining * 0.2)
            cg_time = min(5, remaining * 0.15)
            polish_time = min(2, remaining * 0.05)
        
        try:
            # Phase 1: L-BFGS-B for coarse optimization
            current_pts, current_energy = _lbfgs_optimize(init_pts, time_limit=lbfgs_time)
            
            # Phase 2: Riemannian CG for fine optimization
            if time.time() < deadline - 10:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=cg_time)
                
                # Phase 2b: Another L-BFGS-B after CG
                if time.time() < deadline - 8:
                    current_pts, current_energy = _lbfgs_optimize(current_pts, time_limit=min(5, remaining * 0.1))
            
            # Phase 3: Polish
            if time.time() < deadline - 5:
                current_pts, current_energy = _polish(current_pts, time_limit=polish_time)
            
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
                lambda x: (_log_energy_fast(_normalize(x.reshape(N, 3))), 
                          _log_energy_grad(_normalize(x.reshape(N, 3))).flatten()),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 10000,
                    'ftol': 1e-17,
                    'gtol': 1e-15,
                    'maxls': 100
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
            current_pts, current_energy = _riemannian_cg(current_pts, time_limit=3.0)
        
        # Final tiny-step polish
        if time.time() < deadline - 1:
            current_pts, current_energy = _polish(current_pts, time_limit=2.0)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
