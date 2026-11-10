"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, seed=None):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _perturbed_fibonacci(perturb_scale=0.05, seed=None):
    """Generate perturbed Fibonacci spiral points."""
    base = _fibonacci_spiral()
    rng = np.random.default_rng(seed)
    perturb = rng.normal(0, perturb_scale, (N, 3))
    return _project_to_sphere(base + perturb)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy(pts):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    Using chordal distances on unit sphere: |x_i - x_j| = sqrt(2 - 2*x_i.x_j)
    So log(1/|x_i - x_j|) = -0.5 * log(2 - 2*x_i.x_j)
    """
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    return -0.5 * math.fsum(np.log(vals).tolist())


def _log_energy_and_grad(pts):
    """
    Compute energy and gradient of log(1/|x_i - x_j|) on unit sphere.
    """
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    energy = -0.5 * math.fsum(np.log(vals).tolist())
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = inv_dist_sq @ pts
    
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - grad_dot * pts
    
    return energy, grad


def _optimize_lbfgs(pts, max_iter=2000, ftol=1e-16, gtol=1e-12, time_budget=None):
    """
    L-BFGS optimization on the sphere using retraction (project back to sphere).
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective_and_grad(x):
        if time_budget is not None and time.time() - start > time_budget:
            raise RuntimeError("Time budget exceeded")
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                         'maxls': 50, 'maxfun': 5000})
    pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    return pts_opt, result.fun, result


def _refine(pts, time_budget=5.0):
    """High-precision refinement using adaptive line search."""
    start = time.time()
    pts = pts.copy()
    
    for step in range(300):
        if time.time() - start > time_budget:
            break
        e, g = _log_energy_and_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-15:
            break
        
        # Adaptive line search
        best_e = e
        best_pts = pts.copy()
        best_lr = 0.0
        lr = 0.01
        for _ in range(15):
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
                best_lr = lr
            lr *= 0.5
        
        if np.allclose(best_pts, pts, atol=1e-16):
            break
        pts = best_pts
    
    return pts


def _multi_start_optimize(deadline):
    """
    Multi-start optimization with diverse initializations.
    """
    start_time = time.time()
    
    # Generate diverse starting points
    starts = [
        ("fibonacci", _fibonacci_spiral()),
        ("random1", _random_sphere_points(N, seed=42)),
        ("random2", _random_sphere_points(N, seed=123)),
        ("perturbed_fib1", _perturbed_fibonacci(0.05, seed=42)),
        ("perturbed_fib2", _perturbed_fibonacci(0.1, seed=123)),
    ]
    
    best_pts = None
    best_e = float('inf')
    
    for name, pts in starts:
        if time.time() > deadline - 20.0:
            break
        
        pts = _project_to_sphere(pts)
        
        # Quick L-BFGS optimization with time budget
        time_remaining = deadline - time.time() - 10.0
        if time_remaining < 2.0:
            break
        
        try:
            pts_opt, e_opt, _ = _optimize_lbfgs(
                pts, 
                max_iter=1500, 
                ftol=1e-14, 
                gtol=1e-12,
                time_budget=min(8.0, time_remaining)
            )
            if e_opt < best_e:
                best_e = e_opt
                best_pts = pts_opt
                print(f"  {name}: energy={e_opt:.10f}")
        except Exception as ex:
            print(f"  {name} failed: {ex}")
            continue
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Phase 1: Multi-start optimization
    best_pts, best_e = _multi_start_optimize(deadline)
    
    # If multi-start didn't find anything, fall back to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_spiral()
        best_e = _log_energy(best_pts)
    
    # Phase 2: High-precision refinement
    if time.time() < deadline - 10.0:
        try:
            refined = _refine(best_pts, time_budget=min(15.0, deadline - time.time() - 5.0))
            e_ref = _log_energy(refined)
            if e_ref < best_e:
                best_pts = refined
                best_e = e_ref
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS polish
    if time.time() < deadline - 5.0:
        try:
            time_remaining = deadline - time.time() - 3.0
            if time_remaining > 2.0:
                pts_opt, e_opt, _ = _optimize_lbfgs(
                    best_pts,
                    max_iter=500,
                    ftol=1e-16,
                    gtol=1e-14,
                    time_budget=min(10.0, time_remaining)
                )
                if e_opt < best_e:
                    best_pts = pts_opt
                    best_e = e_opt
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise refinement
    if time.time() < deadline - 2.0:
        try:
            refined2 = _refine(best_pts, time_budget=min(5.0, deadline - time.time() - 1.0))
            e_ref2 = _log_energy(refined2)
            if e_ref2 < best_e:
                best_pts = refined2
                best_e = e_ref2
        except Exception:
            pass
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dot = best_pts @ best_pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, 1)
    min_dist_sq = np.min(dist_sq[iu])
    if min_dist_sq < 1e-18:
        best_pts += np.random.normal(0, 1e-6, (N, 3))
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
