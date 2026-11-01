"""Minimal logarithmic energy of 150 points on the sphere - SA + L-BFGS hybrid."""
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
    # Sum over upper triangle only
    energy = -0.5 * np.sum(log_dists)
    
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


def _simulated_annealing(pts0, max_iter=20000, t_start=0.5, t_end=1e-8, 
                          time_limit=None, start_time=None, seed=42):
    """Simulated annealing on the sphere with geometric cooling."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.default_rng(seed)
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Geometric cooling: T_k = T_start * (T_end/T_start)^(k/max_iter)
    cool_rate = (t_end / t_start) ** (1.0 / max(1, max_iter))
    temperature = t_start
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Generate random tangent direction
        random_vec = rng.standard_normal(3)
        # Project to tangent space at pts[i]
        tangent = random_vec - np.dot(random_vec, pts[i]) * pts[i]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent /= tangent_norm
        
        # Step size proportional to sqrt(temperature)
        step_size = 0.1 * math.sqrt(temperature)
        
        # Move point i along tangent direction
        new_pts = pts.copy()
        new_pts[i] = _exp_map(pts[i:i+1], step_size * tangent[None, :])[0]
        
        # Compute new energy (only need to recompute terms involving point i)
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Metropolis acceptance criterion
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        # Cool down
        temperature *= cool_rate
    
    return best_pts, best_energy


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
            new_pts = _project_to_sphere(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _project_to_sphere(pts + alpha * direction)
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


def _cg_sphere(pts0, max_iter=2000, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient optimization on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # CG state
    p = -grad  # initial search direction
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts  # project to tangent
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search along p using exponential map
        alpha = 0.5
        accepted = False
        for _ in range(50):
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * p):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha = 0.01
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Update CG direction
        g_old = grad
        grad = new_grad
        pts = new_pts
        energy = new_energy
        
        # Fletcher-Reeves restart
        beta = np.sum(grad * grad) / (np.sum(g_old * g_old) + 1e-16)
        p = -grad + beta * p
        p = p - np.sum(p * pts, axis=1, keepdims=True) * pts  # project to tangent
    
    return pts, energy


def _newton_polish(pts0, max_iter=100, time_limit=None, start_time=None, tol=1e-16):
    """Newton polish for final convergence with tighter tolerance."""
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
        for _ in range(30):
            new_pts = _project_to_sphere(pts + alpha * step)
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


def _high_precision_polish(pts0, max_iter=200, time_limit=None, start_time=None, tol=1e-18):
    """Very high precision polish using alternating CG and Newton steps."""
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
        
        # Alternate between CG and Newton-style steps
        if it % 3 == 0:
            # CG step
            p = -grad
            p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
            alpha = 0.1
            accepted = False
            for _ in range(30):
                new_pts = _exp_map(pts, alpha * p)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                pts = new_pts
                grad = new_grad
                energy = new_energy
        else:
            # Newton-style step
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
            
            alpha = 0.5
            accepted = False
            for _ in range(30):
                new_pts = _project_to_sphere(pts + alpha * step)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    accepted = True
                    break
                alpha *= 0.5
            
            if accepted:
                pts = new_pts
                grad = new_grad
                energy = new_energy
            else:
                break
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Simulated annealing with multiple restarts
    # Allocate ~60% of time to SA
    phase1_deadline = start_time + (deadline - start_time) * 0.6
    
    # Different initial configurations for SA
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add random inits
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.03), (5678, 0.08)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    sa_configs = [
        (0.5, 1e-8, 30000),   # High start, low end, many iterations
        (0.3, 1e-8, 25000),
        (0.8, 1e-7, 20000),
        (0.2, 1e-9, 35000),
        (0.6, 1e-6, 15000),
        (0.4, 1e-10, 40000),
    ]
    
    sa_idx = 0
    for idx, init_pts in enumerate(inits):
        if time.time() > phase1_deadline:
            break
        
        remaining = phase1_deadline - time.time()
        # Allocate time based on number of remaining inits
        num_remaining = len(inits) - idx
        time_alloc = min(remaining / num_remaining, 10.0)
        time_alloc = max(time_alloc, 3.0)
        
        # Use different SA configurations
        t_start, t_end, max_iter = sa_configs[sa_idx % len(sa_configs)]
        sa_idx += 1
        
        pts, energy = _simulated_annealing(
            init_pts,
            max_iter=max_iter,
            t_start=t_start,
            t_end=t_end,
            time_limit=time_alloc,
            start_time=time.time(),
            seed=idx * 100 + 42
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: L-BFGS polish on best SA result
    if best_pts is not None and time.time() < deadline - 30:
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
    
    # Phase 3: CG polish
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - time.time() - 10
        if remaining > 3:
            best_pts, best_energy = _cg_sphere(
                best_pts,
                max_iter=1000,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-14
            )
    
    # Phase 4: Newton polish for final convergence
    if best_pts is not None and time.time() < deadline - 8:
        remaining = deadline - time.time() - 5
        if remaining > 2:
            best_pts, best_energy = _newton_polish(
                best_pts,
                max_iter=50,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-16
            )
    
    # Phase 5: High precision polish
    if best_pts is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 1:
            best_pts, best_energy = _high_precision_polish(
                best_pts,
                max_iter=200,
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
