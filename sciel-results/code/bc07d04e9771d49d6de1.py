"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _tangent_grad(points):
    """Compute tangent-space gradient on the sphere."""
    e, g = _energy_grad_fast(points)
    g_tangent = g - (g * points).sum(axis=1, keepdims=True) * points
    return e, g_tangent


def _optimize_single(points, time_budget, maxiter=50000, ftol=1e-14, gtol=1e-12):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    Uses explicit time deadline to stop before exceeding budget.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    p = _project_to_sphere(points.copy())
    best_energy, _ = _tangent_grad(p)
    best_points = p.copy()
    
    # Adaptive step size
    step = 0.01
    min_step = 1e-8
    max_step = 0.5
    
    for it in range(maxiter):
        # Check time deadline with safety margin
        if time.time() > deadline - 0.5:
            break
        
        e, g_tan = _tangent_grad(p)
        
        # Convergence check
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < gtol:
            break
        if abs(best_energy - e) < ftol:
            break
        
        # Backtracking line search
        found = False
        for ls_iter in range(30):
            # Riemannian step: move in tangent direction, then project back
            p_new = p + step * g_tan
            p_new = _project_to_sphere(p_new)
            
            e_new, _ = _tangent_grad(p_new)
            
            if e_new < e - 1e-4 * step * grad_norm * np.linalg.norm(g_tan):
                found = True
                break
            step *= 0.5
            if step < min_step:
                break
        
        if not found:
            # If no step improved, try smaller steps or break
            if step < min_step * 2:
                break
            # Try a very small step
            p_new = p + min_step * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, _ = _tangent_grad(p_new)
            if e_new >= e:
                break
        
        p = p_new
        e = e_new
        if e < best_energy:
            best_energy = e
            best_points = p.copy()
        
        # Increase step if we took a full step
        if found and ls_iter == 0:
            step = min(step * 1.2, max_step)
    
    return best_points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)


def _kick_and_optimize(points, time_budget, kick_scale=0.1):
    """Perturb points and optimize. Returns best of original and kicked."""
    noise = np.random.randn(N, 3) * kick_scale
    kicked_points = _project_to_sphere(points + noise)
    
    budget_kick = time_budget * 0.4
    kicked_opt = _optimize_single(kicked_points, budget_kick, maxiter=20000, ftol=1e-13, gtol=1e-11)
    
    e_kicked = _energy(kicked_opt)
    e_orig = _energy(points)
    
    if e_kicked < e_orig:
        return kicked_opt, e_kicked
    else:
        return points, e_orig


def construct():
    start_time = time.time()
    
    # Phase 1: Fibonacci Spiral Optimization
    points_fib = _fibonacci_spiral(N)
    best_points = points_fib.copy()
    best_energy = _energy(best_points)
    
    # Optimize Fibonacci spiral with tight tolerances
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.25:
        budget = min(TIME_LIMIT * 0.2, TIME_LIMIT * 0.25 - elapsed)
        opt_fib = _optimize_single(points_fib, budget, maxiter=50000, ftol=1e-14, gtol=1e-12)
        e_fib = _energy(opt_fib)
        if e_fib < best_energy:
            best_energy = e_fib
            best_points = opt_fib
    
    # Phase 2: Random Restarts
    np.random.seed(42)
    restarts_done = 0
    max_restarts = 50  # Upper bound, limited by time
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.75:
            break
        
        time_left = TIME_LIMIT * 0.75 - elapsed
        restarts_done += 1
        
        # Adaptive budget: more time per restart when we have more time
        budget_per_restart = min(3.0, time_left * 0.7)
        if budget_per_restart < 0.5:
            break
            
        random_points = _random_points(N)
        opt_rand = _optimize_single(random_points, budget_per_restart, maxiter=20000, ftol=1e-13, gtol=1e-11)
        
        e_rand = _energy(opt_rand)
        if e_rand < best_energy:
            best_energy = e_rand
            best_points = opt_rand
        
        if restarts_done >= max_restarts:
            break

    # Phase 3: Final Polish - high precision
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.92:
        remaining = TIME_LIMIT * 0.92 - elapsed
        if remaining > 2.0:
            final_opt = _optimize_single(best_points, remaining, maxiter=100000, ftol=1e-15, gtol=1e-14)
            e_final = _energy(final_opt)
            if e_final < best_energy:
                best_energy = e_final
                best_points = final_opt

    # Phase 4: Kick and Re-optimize
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.98:
        remaining = TIME_LIMIT * 0.98 - elapsed
        if remaining > 3.0:
            for kick_scale in [0.03, 0.05, 0.1, 0.15]:
                if time.time() - start_time > TIME_LIMIT * 0.98:
                    break
                kicked, e_kicked = _kick_and_optimize(best_points, remaining * 0.3, kick_scale)
                if e_kicked < best_energy:
                    best_energy = e_kicked
                    best_points = kicked
                    if time.time() - start_time < TIME_LIMIT * 0.98:
                        rem = TIME_LIMIT * 0.98 - time.time()
                        if rem > 1.5:
                            new_opt = _optimize_single(best_points, rem, maxiter=50000, ftol=1e-15, gtol=1e-14)
                            e_new = _energy(new_opt)
                            if e_new < best_energy:
                                best_energy = e_new
                                best_points = new_opt

    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
                
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
