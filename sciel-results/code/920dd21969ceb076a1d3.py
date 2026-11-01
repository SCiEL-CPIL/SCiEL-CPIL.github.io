"""Minimal logarithmic energy of 150 points on the sphere - Basin hopping + CG + L-BFGS."""
import time
import math
import numpy as np

N = 150


def _energy_and_grad(pts):
    """Vectorized energy and gradient computation on the sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Check for too-close points
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    if np.any(d_upper < 1e-12):
        return float('inf'), np.zeros_like(pts)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    log_dists = np.log(dists)
    log_dists[iu] = log_dists[iu]  # keep upper triangle
    # Sum over upper triangle only
    energy = -0.5 * np.sum(log_dists)  # since symmetric, sum all / 2
    
    # Gradient: grad_i E = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) / d_ij^2 = sum_j x_j / d_ij^2 - x_i * sum_j 1/d_ij^2
    weighted_sum = inv_d2 @ pts  # (N, 3)
    sum_inv = np.sum(inv_d2, axis=1)  # (N,)
    grad = weighted_sum - pts * sum_inv[:, np.newaxis]
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere."""
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    return pts * cos_t + v * (sin_t / safe_norms)


def _cg_sphere(pts0, max_iter=2000, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient optimization on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Flatten for CG operations
    def flatten(g):
        return g.reshape(-1)
    
    def unflatten(v):
        return v.reshape(N, 3)
    
    def project_tangent(v):
        """Project vector to tangent space at each point."""
        dots = np.sum(v * pts, axis=1)
        return v - dots[:, np.newaxis] * pts
    
    # Initialize CG
    d = -project_tangent(grad.copy())
    d_flat = flatten(d)
    d_norm = np.linalg.norm(d_flat)
    if d_norm < 1e-15:
        return pts, energy
    d /= d_norm
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Line search along d
        alpha = 1.0
        best_alpha = 0.0
        best_energy_ls = energy
        best_pts_ls = pts
        best_grad_ls = grad
        
        for ls in range(30):
            new_pts = _exp_map(pts, alpha * d)
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy_ls:
                best_energy_ls = new_energy
                best_pts_ls = new_pts
                best_grad_ls = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy < energy + 1e-4 * alpha * np.sum(flatten(grad) * flatten(d)):
                break
            
            alpha *= 0.5
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha = 1e-4
            new_pts = _exp_map(pts, alpha * d)
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy >= energy:
                break
            best_pts_ls = new_pts
            best_grad_ls = new_grad
            best_energy_ls = new_energy
            best_alpha = alpha
        
        # Compute new gradient projected to tangent space
        new_grad_tan = project_tangent(best_grad_ls.copy())
        old_grad_tan = project_tangent(grad.copy())
        
        # Check convergence
        if np.linalg.norm(flatten(new_grad_tan)) < tol:
            pts = best_pts_ls
            energy = best_energy_ls
            grad = best_grad_ls
            break
        
        # Polak-Ribiere beta
        grad_diff = flatten(new_grad_tan) - flatten(old_grad_tan)
        old_grad_norm_sq = np.sum(flatten(old_grad_tan) ** 2)
        
        if old_grad_norm_sq > 1e-20:
            beta = max(0.0, np.sum(flatten(new_grad_tan) * grad_diff) / old_grad_norm_sq)
        else:
            beta = 0.0
        
        # New direction
        d = -new_grad_tan + beta * d
        d = project_tangent(d)
        d_norm = np.linalg.norm(flatten(d))
        
        if d_norm < 1e-15:
            # Restart with steepest descent
            d = -new_grad_tan
            d = project_tangent(d)
            d_norm = np.linalg.norm(flatten(d))
            if d_norm < 1e-15:
                pts = best_pts_ls
                energy = best_energy_ls
                grad = best_grad_ls
                break
            d /= d_norm
        
        pts = best_pts_ls
        energy = best_energy_ls
        grad = best_grad_ls
    
    return pts, energy


def _lbgfs_sphere(pts0, max_iter=500, m=15, time_limit=None, start_time=None, tol=1e-14):
    """L-BFGS optimization on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    s_list = []
    y_list = []
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = []
            alphas = []
            
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                rho = 1.0 / si_yi if abs(si_yi) > 1e-16 else 0.0
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            last_s = s_list[-1]
            last_y = y_list[-1]
            sy = np.sum(last_s * last_y)
            gamma = np.sum(last_s * last_s) / sy if abs(sy) > 1e-16 else 1.0
            if gamma < 1e-10 or gamma > 1e10:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[i]
                alpha_i = alphas[i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Project to tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = _exp_map(pts, alpha * direction)
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _exp_map(pts, alpha * direction)
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy


def _newton_polish(pts0, max_iter=50, time_limit=None, start_time=None, tol=1e-16):
    """Newton polish for final convergence."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Diagonal Hessian approximation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_d2 = 1.0 / dists_sq
        np.fill_diagonal(inv_d2, 0.0)
        
        hess_diag = np.sum(inv_d2, axis=1)
        hess_diag = np.maximum(hess_diag, 1e-8)
        
        step = -grad / hess_diag[:, np.newaxis]
        dot_step = np.sum(step * pts, axis=1)
        step -= dot_step[:, np.newaxis] * pts
        
        alpha = 1.0
        accepted = False
        for _ in range(20):
            new_pts = _exp_map(pts, alpha * step)
            new_pts = _project_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            break
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
    
    return pts, energy


def _high_precision_polish(pts0, time_limit=None, start_time=None, tol=1e-18):
    """High-precision final polish using very tight tolerances."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Use CG with very tight tolerance
    pts, energy = _cg_sphere(pts, max_iter=5000, time_limit=time_limit, start_time=start_time, tol=tol)
    
    # Then L-BFGS with tight tolerance
    if time_limit is None or (time.time() - start_time) < time_limit * 0.7:
        remaining = (time_limit - (time.time() - start_time)) * 0.3 if time_limit else 10.0
        pts, energy = _lbgfs_sphere(pts, max_iter=200, m=20, time_limit=remaining, start_time=time.time(), tol=tol * 0.1)
    
    # Final Newton polish
    if time_limit is None or (time.time() - start_time) < time_limit * 0.9:
        remaining = (time_limit - (time.time() - start_time)) * 0.1 if time_limit else 2.0
        pts, energy = _newton_polish(pts, max_iter=50, time_limit=remaining, start_time=time.time(), tol=tol * 0.01)
    
    return pts, energy


def _basin_hopping(pts0, n_iterations=50, perturb_scale=0.1, time_limit=None, start_time=None, seed=42):
    """Basin hopping: perturb + CG minimize, accept if better."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.default_rng(seed)
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(n_iterations):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Perturb: add random noise in tangent space
        noise = rng.standard_normal((N, 3)) * perturb_scale
        # Project to tangent space
        dots = np.sum(noise * pts, axis=1)
        noise -= dots[:, np.newaxis] * pts
        perturbed = _project_to_sphere(pts + noise)
        
        # Minimize with CG
        cg_time = 8.0  # seconds per CG minimization
        min_pts, min_energy = _cg_sphere(perturbed, max_iter=1000, time_limit=cg_time, start_time=time.time(), tol=1e-10)
        
        # Accept if better
        if min_energy < best_energy:
            best_energy = min_energy
            best_pts = min_pts.copy()
            pts = min_pts.copy()
        
        # Gradually reduce perturbation
        perturb_scale *= 0.95
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Basin hopping with multiple initializations
    # Allocate ~75% of time to basin hopping
    phase1_deadline = start_time + (deadline - start_time) * 0.75
    
    # Different initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add random inits
    for seed in [42, 123, 456, 789, 1000, 2024]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.03)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    bh_idx = 0
    for idx, init_pts in enumerate(inits):
        if time.time() > phase1_deadline:
            break
        
        remaining = phase1_deadline - time.time()
        # Allocate time based on number of remaining inits
        num_remaining = len(inits) - idx
        time_alloc = min(remaining / num_remaining, 15.0)
        time_alloc = max(time_alloc, 5.0)
        
        # Basin hopping with different seeds and scales
        perturb_scale = 0.15 if idx % 2 == 0 else 0.1
        n_iter = 40 if time_alloc > 10 else 25
        
        pts, energy = _basin_hopping(
            init_pts,
            n_iterations=n_iter,
            perturb_scale=perturb_scale,
            time_limit=time_alloc,
            start_time=time.time(),
            seed=idx * 100 + 42
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: L-BFGS polish on best basin hopping result
    if best_pts is not None and time.time() < deadline - 25:
        remaining = deadline - time.time() - 20
        if remaining > 5:
            best_pts, best_energy = _lbgfs_sphere(
                best_pts,
                max_iter=300,
                m=20,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-14
            )
    
    # Phase 3: High-precision final polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            best_pts, best_energy = _high_precision_polish(
                best_pts,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-18
            )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Verify validity
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
