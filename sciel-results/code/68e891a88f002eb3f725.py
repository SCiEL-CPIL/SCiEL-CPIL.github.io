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
    """
    High-precision refinement using adaptive line search with best-so-far tracking.
    This is faster than strict Armijo because it allows temporary increases if a later step improves.
    """
    start = time.time()
    pts = pts.copy()
    best_e = _log_energy(pts)
    last_lr = 0.01
    
    for step in range(1000):
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
        # Try a range of step sizes
        lr = max(last_lr * 2.0, 0.01)
        
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
    """
    Final ultra-precise polish with very tight tolerances and small steps.
    """
    start = time.time()
    pts = pts.copy()
    
    for step in range(200):
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse starting points
    starts = [
        ("fibonacci", _fibonacci_spiral()),
        ("random1", _random_sphere_points(N, seed=42)),
        ("random2", _random_sphere_points(N, seed=123)),
        ("random3", _random_sphere_points(N, seed=456)),
        ("antipodal1", _antipodal_pairs(seed=1)),
        ("antipodal2", _antipodal_pairs(seed=2)),
        ("clustered1", _clustered_start(seed=3)),
        ("perturbed_fib1", _perturbed_fibonacci(perturb_scale=0.05, seed=10)),
        ("perturbed_fib2", _perturbed_fibonacci(perturb_scale=0.1, seed=20)),
    ]
    
    best_pts = None
    best_e = float('inf')
    
    # Phase 1: Multi-start L-BFGS
    # Allocate time dynamically. We want to try all starts if possible, but not exceed deadline.
    # Assume each start takes ~10-15s for L-BFGS.
    # We have ~100s for Phase 1, leaving ~45s for refinement/polish.
    
    for idx, (name, init_pts) in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # If we've used too much time, stop starting new optimizations
        if remaining < 20.0:
            break
        
        # Allocate time for this start, but don't exceed remaining time
        # Leave at least 20s for refinement
        time_for_start = min(15.0, remaining - 20.0)
        
        if time_for_start < 2.0:
            break
        
        init_pts = _project_to_sphere(init_pts)
        
        try:
            pts_opt, e_opt, _ = _optimize_lbfgs(init_pts, max_iter=3000, ftol=1e-16, gtol=1e-14,
                                                 time_budget=time_for_start)
            
            if e_opt < best_e:
                best_e = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spiral()
        best_e = _log_energy(best_pts)
    
    # Check time
    if time.time() > deadline - 25.0:
        return _project_to_sphere(best_pts)
    
    # Phase 2: High-precision refinement
    time_for_refine = min(20.0, deadline - time.time() - 15.0)
    if time_for_refine > 3.0:
        try:
            refined = _refine(best_pts, time_budget=time_for_refine)
            e_ref = _log_energy(refined)
            if e_ref < best_e:
                best_pts = refined
                best_e = e_ref
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 15.0:
        return _project_to_sphere(best_pts)
    
    # Phase 3: Second round of L-BFGS from refined point
    time_for_lbfgs2 = min(15.0, deadline - time.time() - 10.0)
    if time_for_lbfgs2 > 3.0:
        try:
            pts_opt2, e_opt2, _ = _optimize_lbfgs(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-15,
                                                   time_budget=time_for_lbfgs2)
            if e_opt2 < best_e:
                best_pts = pts_opt2
                best_e = e_opt2
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 10.0:
        return _project_to_sphere(best_pts)
    
    # Phase 4: Final high-precision polish
    time_for_final = min(10.0, deadline - time.time() - 5.0)
    if time_for_final > 2.0:
        try:
            refined2 = _high_precision_polish(best_pts, time_budget=time_for_final)
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
    if min_dist_sq < 1e-18:  # distance < 1e-9
        # Perturb slightly
        rng = np.random.default_rng(12345)
        best_pts += rng.normal(0, 1e-6, (N, 3))
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
