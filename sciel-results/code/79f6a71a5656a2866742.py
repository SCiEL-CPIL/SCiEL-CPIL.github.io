"""
Optimized approach for logarithmic energy minimization of 150 points on the sphere.
Strategy:
1. Diverse initialization (Fibonacci, Sobol, Random, Jittered).
2. Parallel-like sequential optimization of multiple starts to identify good basins.
3. Aggressive Basin Hopping on the best candidate to escape local minima.
4. High-precision Riemannian polish (L-BFGS with tight tolerances).
"""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    # Use math.fsum for highest precision on the sum
    return -math.fsum(np.log(dist[iu]).tolist())

def _energy_fast(pts):
    """Fast energy calculation using np.sum for optimization loops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_gradient(pts):
    """Energy and gradient (in ambient R^3) of log energy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    # Gradient of -log(dist) is -1/dist * d(dist)/dx
    # d(dist)/dx_i = (x_i - x_j) / dist
    # So grad_i = sum_{j != i} (x_j - x_i) / dist^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    # grad[i] = sum_j (x_j - x_i) / dist^2_ij = - sum_j (x_i - x_j) / dist^2_ij
    # diff[i,j] = x_i - x_j
    # So grad = - sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms

def _fibonacci_points(n):
    """Fibonacci spiral (low-discrepancy) points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_points(n, seed=12345):
    """Quasi-random (Sobol) points on unit sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random(n)
    pts = 2.0 * raw - 1.0
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms

def _random_points(n, seed=42):
    """Random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return _project_to_sphere(pts)

def _jittered_fibonacci(n, jitter=0.1, seed=42):
    """Fibonacci points with random jitter to break symmetry."""
    rng = np.random.default_rng(seed)
    pts = _fibonacci_points(n)
    noise = rng.normal(0, jitter, pts.shape)
    return _project_to_sphere(pts + noise)

def _optimize_on_sphere(pts, deadline, maxiter=2000, ftol=1e-14, gtol=1e-12):
    """Optimize log energy on the sphere using L-BFGS-B with tangent gradient."""
    pts = _project_to_sphere(pts.copy())

    def fun_and_grad(flat):
        p = flat.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        # Project gradient to tangent space
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tan.flatten()

    x0 = pts.flatten()
    best = pts.copy()
    best_e = _log_energy(pts)
    try:
        res = minimize(
            fun_and_grad, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                     'maxls': 50, 'maxcor': 40}
        )
        p1 = _project_to_sphere(res.x.reshape(N, 3))
        e1 = _log_energy(p1)
        if e1 < best_e:
            best, best_e = p1, e1
    except Exception:
        pass
    
    # If we have time left, do a second pass with tighter tolerances
    if time.time() < deadline - 2.0 and best_e > -1e6:
        try:
            res2 = minimize(
                fun_and_grad, best.flatten(), method='L-BFGS-B', jac=True,
                options={'maxiter': max(500, maxiter // 2), 'ftol': ftol * 0.1, 'gtol': gtol * 0.1,
                         'maxls': 50, 'maxcor': 40}
            )
            p2 = _project_to_sphere(res2.x.reshape(N, 3))
            e2 = _log_energy(p2)
            if e2 < best_e:
                best, best_e = p2, e2
        except Exception:
            pass
    
    return best, best_e

def _tangent_grad_descent_polish(pts, deadline, max_iter=10000, base_lr=1e-3):
    """High-precision line-search gradient descent on the sphere (Riemannian)."""
    pts = _project_to_sphere(pts.copy())
    best = pts.copy()
    best_e = _log_energy(pts)
    current_e = best_e
    lr = base_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, g = _energy_and_gradient(pts)
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        gnorm = np.linalg.norm(g_tan)
        if gnorm < 1e-14:
            break
        
        # Line search (backtracking Armijo)
        alpha = lr
        improved = False
        for _ in range(40):
            cand = pts - alpha * g_tan
            cand = _project_to_sphere(cand)
            ce = _energy_fast(cand)
            if ce < current_e - 1e-4 * alpha * gnorm * gnorm:
                pts = cand
                current_e = ce
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
            continue
        
        if current_e < best_e - 1e-15:
            best = pts.copy()
            best_e = current_e
        
        # Occasionally increase learning rate if stuck
        if it % 100 == 0 and lr < base_lr:
            lr = base_lr * 0.1
    
    return best, best_e

def _basin_hopping(pts, deadline, max_iters=40):
    """Basin hopping with simulated annealing acceptance to escape local minima."""
    current_pts = pts.copy()
    current_energy = _log_energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    # Simulated annealing parameters
    T_initial = 0.5
    T_min = 0.001
    cooling_rate = 0.95
    
    # Track consecutive non-improvements to adapt noise scale
    stuck_count = 0
    base_scale = 0.05
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Adaptive temperature
        T = max(T_min, T_initial * (cooling_rate ** i))
        
        # Adaptive noise scale: increase when stuck, decrease when improving
        if stuck_count > 5:
            scale = min(0.3, base_scale * (1.0 + stuck_count * 0.1))
        elif stuck_count > 0:
            scale = base_scale * (1.0 + stuck_count * 0.05)
        else:
            scale = base_scale * 0.7
        
        noise = rng.normal(0, scale, current_pts.shape)
        perturbed = _project_to_sphere(current_pts + noise)
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        alloc = max(2.0, min(12.0, remaining * 0.15))
        
        opt_pts, opt_e = _optimize_on_sphere(perturbed, time.time() + alloc, maxiter=1000)
        
        # Simulated annealing acceptance
        delta_e = opt_e - current_energy
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            current_pts = opt_pts
            current_energy = opt_e
            stuck_count = 0
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
        else:
            stuck_count += 1
        
        # Periodically restart from best to avoid drifting too far
        if i % 10 == 9 and stuck_count > 3:
            current_pts = best_pts.copy()
            current_energy = best_energy
            stuck_count = 0

    return best_pts, best_energy

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multiple diverse starting configurations
    # We generate 6 diverse starts to increase the chance of finding a good basin
    start_configs = [
        _fibonacci_points(N),
        _sobol_points(N, seed=12345),
        _sobol_points(N, seed=99999),
        _random_points(N, seed=1),
        _jittered_fibonacci(N, jitter=0.1, seed=2),
        _jittered_fibonacci(N, jitter=0.2, seed=3),
    ]
    
    # Optimize each starting config with a time budget
    # We allocate roughly 15s per config, but adjust based on remaining time
    # Prioritize configurations that are likely to be good (low initial energy)
    time_per_config = 12.0
    for i, cfg in enumerate(start_configs):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        alloc = min(time_per_config, remaining - 40.0) # Keep 40s for BH and polish
        if alloc < 3.0:
            break
        
        # Quick initial optimization to estimate quality
        opt_pts, opt_e = _optimize_on_sphere(cfg, time.time() + alloc, maxiter=1200)
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts.copy()
            # If we found a significantly better config, spend more time on it
            if i > 0 and best_energy < -2360:
                # Do an additional optimization pass on the best config
                extra_alloc = min(5.0, deadline - time.time() - 40.0)
                if extra_alloc > 2.0:
                    opt_pts2, opt_e2 = _optimize_on_sphere(best_pts, time.time() + extra_alloc, maxiter=1000)
                    if opt_e2 < best_energy:
                        best_energy = opt_e2
                        best_pts = opt_pts2.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_points(N)
        best_energy = _log_energy(best_pts)
    
    # Phase 2: Basin hopping to escape local minima
    # Use a significant portion of the remaining time for this
    if time.time() < deadline - 50.0:
        bh_deadline = deadline - 25.0
        best_pts, best_energy = _basin_hopping(best_pts, bh_deadline, max_iters=50)
    
    # Phase 3: Polish with gradient descent
    if time.time() < deadline - 15.0:
        p, e = _tangent_grad_descent_polish(best_pts, deadline - 8.0, max_iter=20000, base_lr=1e-3)
        if e < best_energy:
            best_energy = e
            best_pts = p
    
    # Final L-BFGS tight pass with multiple restarts if time permits
    if time.time() < deadline - 8.0:
        # Try a few different tight passes with slightly different tolerances
        for ftol_val, gtol_val in [(1e-16, 1e-14), (1e-18, 1e-15), (1e-15, 1e-13)]:
            if time.time() > deadline - 5.0:
                break
            p, e = _optimize_on_sphere(best_pts, deadline - 3.0, maxiter=5000, ftol=ftol_val, gtol=gtol_val)
            if e < best_energy:
                best_energy = e
                best_pts = p
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9, finite values
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        best_pts = _project_to_sphere(_fibonacci_points(N))
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-9:
        best_pts = _project_to_sphere(_fibonacci_points(N))
    
    return best_pts
# EVOLVE-BLOCK-END
