"""Minimal logarithmic energy of 150 points on the sphere - Combined Riemannian + SciPy L-BFGS-B."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
MARGIN = 8.0

def _energy_grad(pts):
    """
    Vectorized logarithmic energy and gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (R3) = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Returns (energy, grad) where grad is (N,3) in R3 (caller projects to tangent).
    """
    n = N
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)
    
    iu = np.triu_indices(n, 1)
    d_upper = d[iu]
    if np.any(d_upper < 1e-14):
        return float('inf'), np.zeros((n, 3))
    
    terms = np.log(1.0 / d_upper)
    energy = math.fsum(terms.tolist())
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    col_sum = np.sum(inv_d2, axis=1)
    weighted = inv_d2 @ pts
    grad = pts * col_sum[:, None] - weighted
    
    return energy, grad

def _project(pts):
    """Project to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _exp_map(pts, v):
    """Exponential map: move along geodesic by tangent vector v."""
    dot = np.sum(pts * v, axis=1, keepdims=True)
    v_t = v - dot * pts
    
    vnorm = np.linalg.norm(v_t, axis=1, keepdims=True)
    vnorm_safe = np.where(vnorm < 1e-15, 1.0, vnorm)
    
    cos_t = np.cos(vnorm_safe)
    sin_t = np.sin(vnorm_safe) / vnorm_safe
    sin_t = np.where(vnorm < 1e-15, 0.0, sin_t)
    
    new_pts = pts * cos_t + v_t * sin_t
    return _project(new_pts)

def _retract(pts, v):
    """Retraction: project pts + v to sphere."""
    return _project(pts + v)

def _fibonacci_sphere(n):
    """Fibonacci spiral initialization."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)

def _halton_sphere(n):
    """Halton sequence initialization."""
    def halton(idx, base):
        result = 0.0
        f = 1.0
        i = idx
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    pts = []
    for i in range(1, n + 1):
        x, y, z = halton(i, 2), halton(i, 3), halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        pts.append(p / norm if norm > 1e-15 else np.array([1.0, 0.0, 0.0]))
    return np.array(pts)

def _sobol_sphere(n, seed=0):
    """Sobol-like quasi-random initialization using structured jitter."""
    rng = np.random.default_rng(seed)
    z_vals = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0/n
    z_vals = (z_vals - 1.0) * 2.0 - 1.0
    z_vals = np.clip(z_vals, -1.0 + 1e-6, 1.0 - 1e-6)
    z_vals = z_vals + rng.uniform(-0.3, 0.3, n)
    z_vals = np.clip(z_vals, -1.0 + 1e-6, 1.0 - 1e-6)
    
    phi_vals = 2.0 * np.pi * (np.arange(n) * (1.0 + np.sqrt(5.0)) / 2.0) % (2.0 * np.pi)
    phi_vals = phi_vals + rng.uniform(-0.1, 0.1, n)
    
    r = np.sqrt(np.maximum(0.0, 1.0 - z_vals * z_vals))
    pts = np.column_stack([r * np.cos(phi_vals), r * np.sin(phi_vals), z_vals])
    return _project(pts)

def _scipy_lbfgs(pts0, time_limit, maxiter=2000):
    """
    Use scipy.optimize.minimize with L-BFGS-B on flattened coordinates.
    Normalizes points internally to stay on sphere.
    """
    start = time.time()
    x0 = pts0.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        energy, grad_r3 = _energy_grad(p)
        if energy == float('inf'):
            return 1e10, np.zeros_like(x)
        # Project gradient to tangent
        dot = np.sum(p * grad_r3, axis=1, keepdims=True)
        grad_t = grad_r3 - dot * p
        return energy, grad_t.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 5000, 'maxls': 30})
        x_opt = res.x.reshape(N, 3)
        x_opt = _project(x_opt)
        energy, _ = _energy_grad(x_opt)
        return x_opt, energy
    except Exception:
        return pts0, _energy_grad(pts0)[0]

def _riemannian_cg(pts0, time_limit, maxiter=3000, tol=1e-13):
    """Riemannian conjugate gradient on sphere."""
    start = time.time()
    n = N
    
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    gnorm = np.linalg.norm(grad_t)
    
    if gnorm < tol:
        return pts, energy
    
    d = -grad_t / gnorm
    alpha = 0.5
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        best_e = energy
        best_p = pts
        best_g = grad_t
        a = alpha
        improved = False
        
        for _ls in range(30):
            new_pts = _exp_map(pts, a * d)
            new_e, new_g_r3 = _energy_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_p = new_pts
                ddot = np.sum(new_pts * new_g_r3, axis=1, keepdims=True)
                best_g = new_g_r3 - ddot * new_pts
                improved = True
                break
            a *= 0.5
        
        if not improved:
            alpha *= 0.1
            if alpha < 1e-14:
                break
            continue
        
        old_gnorm = gnorm
        new_gnorm = np.linalg.norm(best_g)
        
        if new_gnorm < tol:
            pts, energy = best_p, best_e
            break
        
        beta = np.sum(best_g * (best_g - grad_t)) / (old_gnorm * old_gnorm + 1e-15)
        beta = max(0.0, beta)
        
        d = -best_g + beta * d * (old_gnorm / (new_gnorm + 1e-15))
        
        pts, energy, grad_t = best_p, best_e, best_g
        gnorm = new_gnorm
        alpha = min(0.5, alpha * 1.2)
    
    return pts, energy

def _custom_lbfgs(pts0, time_limit, maxiter=2000, m=10, tol=1e-13):
    """
    Custom Riemannian L-BFGS with two-loop recursion on the sphere.
    """
    start = time.time()
    n = N
    
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < tol:
        return pts, energy
    
    s_list = []
    y_list = []
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < tol:
            break
        
        if len(s_list) == 0:
            direction = -grad_t
        else:
            q = grad_t.copy()
            rhos = []
            alphas = []
            
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rho = 0.0
                else:
                    rho = 1.0 / si_yi
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            last_s = s_list[-1]
            last_y = y_list[-1]
            sy = np.sum(last_s * last_y)
            if abs(sy) > 1e-16:
                gamma = np.sum(last_s * last_s) / sy
                if gamma < 1e-10 or gamma > 1e10:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        alpha = step_size
        accepted = False
        grad_dir = np.sum(grad_t * direction)
        
        for _ in range(50):
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy == float('inf'):
                alpha *= 0.5
                continue
            
            dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
            new_grad_t = new_grad_r3 - dot_new * new_pts
            
            if new_energy < energy + 1e-4 * alpha * grad_dir:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy < energy and new_energy != float('inf'):
                dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
                new_grad_t = new_grad_r3 - dot_new * new_pts
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        s = alpha * direction
        y = new_grad_t - grad_t
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad_t = new_grad_t
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy

def _high_precision_polish(pts0, time_limit, maxiter=500):
    """
    High-precision final polish using very small steps and tight tolerances.
    Combines gradient descent with occasional CG restarts.
    """
    start = time.time()
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < 1e-15:
        return pts, energy
    
    step_size = 1e-4
    min_step = 1e-16
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-15:
            break
        
        # Gradient descent step
        direction = -grad_t / (gnorm + 1e-15)
        
        alpha = step_size
        accepted = False
        
        for _ in range(40):
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy == float('inf'):
                alpha *= 0.5
                continue
            
            if new_energy < energy:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
            continue
        
        dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
        new_grad_t = new_grad_r3 - dot_new * new_pts
        
        pts = new_pts
        energy = new_energy
        grad_t = new_grad_t
        step_size = min(alpha * 1.3, 0.1)
        
        # Every 50 iterations, try a small CG burst
        if it % 50 == 49:
            remaining = time_limit - (time.time() - start)
            if remaining > 2.0:
                pts, energy = _riemannian_cg(pts, min(2.0, remaining * 0.3), maxiter=50, tol=1e-14)
                energy, grad_r3 = _energy_grad(pts)
                dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
                grad_t = grad_r3 - dot * pts
    
    return pts, energy

def _optimize(pts, time_limit):
    """
    Multi-stage optimization:
    1. Riemannian CG for fast descent
    2. SciPy L-BFGS-B for robust convergence
    3. Custom Riemannian L-BFGS for fine polish
    4. High-precision final polish
    """
    start = time.time()
    best_pts = pts.copy()
    best_e, _ = _energy_grad(pts)
    
    # Stage 1: CG for fast descent (25% of time)
    t1 = time_limit * 0.25
    pts, e = _riemannian_cg(pts, t1, maxiter=3000, tol=1e-13)
    if e < best_e:
        best_e, best_pts = e, pts.copy()
    
    # Stage 2: SciPy L-BFGS-B (30% of time)
    remaining = time_limit - (time.time() - start)
    if remaining > 5.0:
        t2 = remaining * 0.30
        pts, e = _scipy_lbfgs(pts, t2, maxiter=1000)
        if e < best_e:
            best_e, best_pts = e, pts.copy()
    
    # Stage 3: Custom Riemannian L-BFGS (25% of remaining)
    remaining = time_limit - (time.time() - start)
    if remaining > 5.0:
        t3 = remaining * 0.25
        pts, e = _custom_lbfgs(pts, t3, maxiter=500, m=15, tol=1e-14)
        if e < best_e:
            best_e, best_pts = e, pts.copy()
    
    # Stage 4: High-precision polish (remaining)
    remaining = time_limit - (time.time() - start)
    if remaining > 2.0:
        pts, e = _high_precision_polish(pts, remaining - 0.5, maxiter=500)
        if e < best_e:
            best_e, best_pts = e, pts.copy()
    
    return best_pts, best_e

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_sobol_sphere(N, seed=42))
    
    # Random initializations
    for seed in [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]:
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((N, 3))
        inits.append(_project(pts))
    
    # Perturbed Fibonacci
    for seed, strength in [(3, 0.03), (5, 0.08), (11, 0.15), (23, 0.25)]:
        rng = np.random.default_rng(seed)
        base = _fibonacci_sphere(N)
        pert = base + rng.standard_normal((N, 3)) * strength
        inits.append(_project(pert))
    
    # Ring-based initialization
    n_rings = 12
    pts_ring = []
    for k in range(n_rings):
        z = 1.0 - 2.0 * (k + 0.5) / n_rings
        r = np.sqrt(max(0.0, 1.0 - z * z))
        n_per_ring = N // n_rings + (1 if k < N % n_rings else 0)
        angles = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
        for a in angles:
            pts_ring.append([r * np.cos(a), r * np.sin(a), z])
    if len(pts_ring) == N:
        inits.append(_project(np.array(pts_ring)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization on all inits
    n_inits = len(inits)
    coarse_budget = (DEADLINE - MARGIN) * 0.40
    per_init_time = coarse_budget / n_inits
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - (DEADLINE - MARGIN) * 0.60:
            break
        
        remaining_for_coarse = deadline - (DEADLINE - MARGIN) * 0.60 - time.time()
        if remaining_for_coarse < 2.0:
            break
        
        t_alloc = min(per_init_time, remaining_for_coarse)
        if t_alloc < 1.0:
            break
        
        pts, energy = _optimize(init_pts, t_alloc)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Deep polish on best result with remaining time
    if best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 10.0:
            best_pts, best_energy = _optimize(best_pts, remaining - 2.0)
        
        # Phase 3: Final high-precision polish
        remaining = deadline - time.time()
        if remaining > 5.0:
            pts, e = _high_precision_polish(best_pts, remaining - 1.0, maxiter=300)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    else:
        best_pts = _project(best_pts)
    
    # Validate
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        iu = np.triu_indices(N, 1)
        min_d = np.sqrt(d2[iu]).min()
        if min_d < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
