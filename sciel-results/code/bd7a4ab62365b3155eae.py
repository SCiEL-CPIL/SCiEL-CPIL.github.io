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


def _antipodal_pairs(seed=None):
    """Generate points as antipodal pairs with random orientations."""
    rng = np.random.default_rng(seed)
    n_pairs = N // 2
    pts = rng.normal(size=(n_pairs, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-15)
    # Create antipodal pairs
    all_pts = np.vstack([pts, -pts])
    return all_pts


def _clustered_start(seed=None):
    """Generate clustered starting points for diversity."""
    rng = np.random.default_rng(seed)
    # Create 10 clusters of 10 points
    n_clusters = 10
    pts_per_cluster = N // n_clusters
    cluster_centers = rng.normal(size=(n_clusters, 3))
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / np.maximum(norms, 1e-15)
    
    pts = []
    for center in cluster_centers:
        offsets = rng.normal(0, 0.3, (pts_per_cluster, 3))
        cluster_pts = center + offsets
        norms = np.linalg.norm(cluster_pts, axis=1, keepdims=True)
        cluster_pts = cluster_pts / np.maximum(norms, 1e-15)
        pts.append(cluster_pts)
    return np.vstack(pts)


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
    
    E = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    dE/dx_k = sum_{j!=k} x_j / (2 - 2*x_k.x_j)
    Project to tangent space.
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
    best_pts = pts.copy()
    best_e = float('inf')
    
    def objective_and_grad(x):
        nonlocal best_pts, best_e
        if time_budget is not None and time.time() - start > time_budget:
            raise RuntimeError("Time budget exceeded")
        pts_local = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts_local)
        if e < best_e:
            best_e = e
            best_pts = pts_local.copy()
        return e, g.flatten()
    
    try:
        result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                             'maxls': 50, 'maxfun': 5000})
        pts_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_pts = pts_opt
        return best_pts, best_e, result
    except RuntimeError:
        return best_pts, best_e, None
    except Exception:
        return best_pts, best_e, None


def _refine(pts, time_budget=5.0):
    """High-precision refinement using adaptive line search with better step-size tracking."""
    start = time.time()
    pts = pts.copy()
    best_e = _log_energy(pts)
    last_lr = 0.01
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_and_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-15:
            break
        
        # Adaptive line search with warm-started step size
        best_e_local = e
        best_pts_local = pts.copy()
        best_lr = 0.0
        lr = max(last_lr * 2.0, 0.01)  # Try larger step first if previous worked
        
        for _ in range(25):
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            
            if new_e < best_e_local:
                best_e_local = new_e
                best_pts_local = new_pts
                best_lr = lr
                last_lr = lr
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-16):
            break
        
        pts = best_pts_local
        best_e = best_e_local
    
    return pts


def _high_precision_polish(pts, time_budget=5.0):
    """Final ultra-precise polish with very tight tolerances."""
    start = time.time()
    pts = pts.copy()
    
    for step in range(100):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_and_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-18:
            break
        
        # Very small steps for high precision
        best_e_local = e
        best_pts_local = pts.copy()
        lr = 0.001
        
        for _ in range(30):
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            
            if new_e < best_e_local:
                best_e_local = new_e
                best_pts_local = new_pts
            else:
                lr *= 0.5
                if lr < 1e-15:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-18):
            break
        
        pts = best_pts_local
    
    return pts


def _multi_start_optimize(deadline):
    """
    Multi-start optimization with diverse initializations and time-aware scheduling.
    """
    start_time = time.time()
    
    # Generate diverse starting points
    starts = [
        ("fibonacci", _fibonacci_spiral()),
        ("random1", _random_sphere_points(N, seed=42)),
        ("random2", _random_sphere_points(N, seed=123)),
        ("random3", _random_sphere_points(N, seed=456)),
        ("perturbed_fib1", _perturbed_fibonacci(0.05, seed=42)),
        ("perturbed_fib2", _perturbed_fibonacci(0.1, seed=123)),
        ("antipodal", _antipodal_pairs(seed=42)),
        ("clustered", _clustered_start(seed=42)),
    ]
    
    best_pts = None
    best_e = float('inf')
    
    # Allocate time: 60% for multi-start, 30% for refinement, 10% for polish
    multi_start_budget = 0.6 * (deadline - start_time)
    per_start_budget = multi_start_budget / len(starts)
    
    for name, pts in starts:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - 15.0  # Reserve 15s for refinement+polish
        if remaining < 3.0:
            break
        
        pts = _project_to_sphere(pts)
        
        # Quick L-BFGS optimization with time budget
        time_for_this = min(per_start_budget, remaining - 10.0)
        if time_for_this < 2.0:
            break
        
        try:
            pts_opt, e_opt, _ = _optimize_lbfgs(
                pts, 
                max_iter=1500, 
                ftol=1e-14, 
                gtol=1e-12,
                time_budget=time_for_this
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
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Phase 1: Multi-start optimization (60% of time)
    best_pts, best_e = _multi_start_optimize(deadline)
    
    # If multi-start didn't find anything, fall back to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_spiral()
        best_e = _log_energy(best_pts)
    
    # Phase 2: High-precision refinement (30% of time)
    if time.time() < deadline - 10.0:
        try:
            time_for_refine = min(30.0, deadline - time.time() - 8.0)
            if time_for_refine > 3.0:
                refined = _refine(best_pts, time_budget=time_for_refine)
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
            if time_remaining > 3.0:
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
    
    # Phase 4: Final ultra-precise refinement (10% of time)
    if time.time() < deadline - 2.0:
        try:
            time_for_polish = min(8.0, deadline - time.time() - 1.0)
            if time_for_polish > 2.0:
                refined2 = _high_precision_polish(best_pts, time_budget=time_for_polish)
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
    
    # Final safety check: ensure all values are finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
