"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # We need to be careful with the diagonal.
    # inv_dist_sq is 1/d^2. We set diagonal to 0 so it doesn't contribute.
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    # Avoid log(0) or log(negative) though d_sq should be positive
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


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    # Project gradient to tangent space: g_tangent = g - (g·n)n where n = p (unit normal)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _optimize_single(points, time_budget, maxiter=3000, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization from a single starting point."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        # Use a custom wrapper to check time inside the optimizer? 
        # Scipy's minimize doesn't support time limits directly.
        # We rely on maxiter and the fact that each iteration is fast.
        # If it takes too long, we will just return the last best point found by the wrapper?
        # Actually, minimize returns the best point found so far if it stops.
        # But if it hangs, we are stuck. 
        # For N=100, one iteration is ~1-5ms. 3000 iters ~ 15s. 
        # We set maxiter based on time_budget roughly.
        
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


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
    # Create a kicked version
    noise = np.random.randn(N, 3) * kick_scale
    kicked_points = _project_to_sphere(points + noise)
    
    # Optimize the kicked version
    # Use a fraction of the time budget
    t_kick = time.time()
    budget_kick = time_budget * 0.5
    kicked_opt = _optimize_single(kicked_points, budget_kick, maxiter=2000)
    
    e_kicked = _energy(kicked_opt)
    e_orig = _energy(points)
    
    if e_kicked < e_orig:
        return kicked_opt, e_kicked
    else:
        return points, e_orig


def construct():
    start_time = time.time()
    
    # Phase 1: Fibonacci Spiral Optimization
    # This is usually a very good starting point.
    points_fib = _fibonacci_spiral(N)
    best_points = points_fib.copy()
    best_energy = _energy(best_points)
    
    # Optimize Fibonacci spiral with tight tolerances
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.3:
        budget = min(TIME_LIMIT * 0.25, TIME_LIMIT * 0.3 - elapsed)
        opt_fib = _optimize_single(points_fib, budget, maxiter=5000, ftol=1e-14, gtol=1e-12)
        e_fib = _energy(opt_fib)
        if e_fib < best_energy:
            best_energy = e_fib
            best_points = opt_fib
    
    # Phase 2: Random Restarts
    # We will run several random restarts.
    # Each restart: Generate random points, optimize with moderate settings.
    np.random.seed(42)
    restarts_done = 0
    max_restarts = 20 # Upper bound, limited by time
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.7:
            break
        
        # Estimate time per restart
        # If we have 40s left and want 10 restarts, 4s each.
        time_left = TIME_LIMIT * 0.7 - elapsed
        restarts_done += 1
        
        # Adaptive budget: if we are early, use larger budget. If late, smaller.
        # Let's just use a fixed reasonable budget per restart, e.g., 3-5 seconds.
        # But ensure we don't exceed the phase limit.
        budget_per_restart = min(5.0, time_left * 0.8)
        if budget_per_restart < 1.0:
            break
            
        random_points = _random_points(N)
        
        # Optimize random points
        # Use moderate iterations first
        opt_rand = _optimize_single(random_points, budget_per_restart, maxiter=2000, ftol=1e-12, gtol=1e-10)
        
        e_rand = _energy(opt_rand)
        if e_rand < best_energy:
            best_energy = e_rand
            best_points = opt_rand
            
        # If this was a good start, maybe refine it a bit more?
        # No, let's keep it simple and rely on the final polish.
        
        if restarts_done >= max_restarts:
            break

    # Phase 3: Final Polish
    # Take the best point found so far and run a very high-precision optimization.
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.9:
        remaining = TIME_LIMIT * 0.9 - elapsed
        if remaining > 2.0:
            # Use a lot of iterations and very tight tolerances
            final_opt = _optimize_single(best_points, remaining, maxiter=10000, ftol=1e-15, gtol=1e-14)
            e_final = _energy(final_opt)
            if e_final < best_energy:
                best_energy = e_final
                best_points = final_opt

    # Phase 4: Kick and Re-optimize (if time allows)
    # Try to escape the current local minimum by kicking the points.
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.98:
        remaining = TIME_LIMIT * 0.98 - elapsed
        if remaining > 3.0:
            # Try a few kicks with different scales
            for kick_scale in [0.05, 0.1, 0.2]:
                if time.time() - start_time > TIME_LIMIT * 0.98:
                    break
                kicked, e_kicked = _kick_and_optimize(best_points, remaining * 0.4, kick_scale)
                if e_kicked < best_energy:
                    best_energy = e_kicked
                    best_points = kicked
                    # If we found a better one, we might want to polish it again?
                    # The kick_and_optimize already optimizes it.
                    # But maybe a final tight polish on the new best?
                    if time.time() - start_time < TIME_LIMIT * 0.98:
                        rem = TIME_LIMIT * 0.98 - time.time()
                        if rem > 1.0:
                            new_opt = _optimize_single(best_points, rem, maxiter=5000, ftol=1e-14, gtol=1e-12)
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
                # Perturb one of them
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
                
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci if something went wrong
        best_points = _fibonacci_spiral(N)
    
    return best_points
