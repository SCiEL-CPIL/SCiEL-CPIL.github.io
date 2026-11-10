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
    # Compute all pairwise dot products
    # pts is (N, 3), so pts @ pts.T is (N, N)
    dot = pts @ pts.T
    # Clip to avoid log of negative or zero
    # 2 - 2*dot should be in (0, 4] for distinct points on sphere
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # avoid self-interaction issues
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    return -0.5 * math.fsum(np.log(vals).tolist())


def _log_energy_and_grad(pts):
    """
    Compute energy and gradient of log(1/|x_i - x_j|) on unit sphere.
    
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    
    For a point x_k, the gradient contribution from pair (i,j) containing k:
    If k=i: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    If k=j: d/dx_j log(1/|x_i - x_j|) = -(x_j - x_i)/|x_i - x_j|^2
    
    But we need the gradient on the sphere, so we project: grad_sphere = grad - (grad.x_k)*x_k
    
    Actually, let's use the formula directly.
    E = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    dE/dx_k = -0.5 * sum_{j!=k} d/dx_k log(2 - 2*x_k.x_j)
            = -0.5 * sum_{j!=k} (-2*x_j)/(2 - 2*x_k.x_j)
            = sum_{j!=k} x_j / (2 - 2*x_k.x_j)
            = sum_{j!=k} x_j / (2*(1 - x_k.x_j))
    
    Then project to tangent: grad_tangent = grad - (grad . x_k) * x_k
    """
    dot = pts @ pts.T
    # dist_sq[i,j] = 2 - 2*dot[i,j]
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    energy = -0.5 * math.fsum(np.log(vals).tolist())
    
    # Gradient: grad_k = sum_{j != k} x_j / dist_sq[k,j]
    # dist_sq is (N, N), symmetric
    # For each k, grad_k = sum_j x_j / dist_sq[k,j] for j != k
    # We can compute this as: (1/dist_sq) @ pts, but with diagonal excluded
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = inv_dist_sq @ pts  # (N, 3)
    
    # Project gradient to tangent space
    # grad_proj = grad - (grad . pts) * pts
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
                                             'maxls': 100, 'maxfun': 10000})
        pts_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        if e_opt < best_e:
            best_pts = pts_opt
            best_e = e_opt
        return best_pts, best_e, result
    except RuntimeError:
        # Time exceeded, return best available
        e = _log_energy(best_pts)
        return best_pts, e, None
    except Exception:
        e = _log_energy(best_pts)
        return best_pts, e, None


def _refine(pts, time_budget=5.0):
    """High-precision refinement using adaptive step size and backtracking line search."""
    start = time.time()
    pts = pts.copy()
    best_e = _log_energy(pts)
    
    # Use a more aggressive initial learning rate and more backtracking steps
    base_lr = 0.5
    for step in range(2000):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_and_grad(pts)
        
        # Check gradient norm
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        
        # Adaptive line search with backtracking (Armijo condition)
        lr = base_lr
        best_e_local = e
        best_pts_local = pts.copy()
        c1 = 1e-4  # Armijo parameter
        
        for _ in range(30):
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            
            # Armijo condition: f(x + lr*g) <= f(x) + c1 * lr * g^T * g
            if new_e <= e + c1 * lr * np.sum(g * g):
                best_e_local = new_e
                best_pts_local = new_pts
                break
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-15):
            break
        
        pts = best_pts_local
        best_e = best_e_local
        # Gradually reduce base learning rate for convergence
        base_lr = max(0.1, base_lr * 0.995)
    
    return pts


def _multi_start_optimize(time_budget_per_start=10.0, num_starts=3):
    """Try multiple initializations and return the best result."""
    start_time = time.time()
    
    # Define different initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral())
    
    # 2. Random points with different seeds
    for seed in [42, 123, 777]:
        initializations.append(_random_sphere_points(N, seed=seed))
    
    best_pts = None
    best_e = float('inf')
    
    for idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = time_budget_per_start * (idx + 1) - elapsed
        if remaining < 1.0:
            break
        
        init_pts = _project_to_sphere(init_pts)
        
        try:
            # L-BFGS optimization with tighter tolerances
            pts_opt, e_opt, _ = _optimize_lbfgs(init_pts, max_iter=3000, ftol=1e-16, gtol=1e-14,
                                                 time_budget=remaining)
            
            if e_opt < best_e:
                best_e = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Phase 1: Multi-start optimization with more starts and tighter tolerances
    best_pts, best_e = _multi_start_optimize(time_budget_per_start=20.0, num_starts=4)
    
    if best_pts is None:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral()
        best_e = _log_energy(best_pts)
    
    # Check time
    if time.time() > deadline - 25.0:
        return _project_to_sphere(best_pts)
    
    # Phase 2: High-precision refinement with Armijo line search
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
    
    # Phase 3: Second round of L-BFGS from refined point with very tight tolerances
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
    
    # Phase 4: Final high-precision refinement
    time_for_final = min(10.0, deadline - time.time() - 5.0)
    if time_for_final > 2.0:
        try:
            refined2 = _refine(best_pts, time_budget=time_for_final)
            e_ref2 = _log_energy(refined2)
            if e_ref2 < best_e:
                best_pts = refined2
                best_e = e_ref2
        except Exception:
            pass
    
    # Phase 5: One more L-BFGS polish if time permits
    if time.time() < deadline - 8.0:
        time_for_polish = min(8.0, deadline - time.time() - 3.0)
        if time_for_polish > 2.0:
            try:
                pts_polish, e_polish, _ = _optimize_lbfgs(best_pts, max_iter=2000, ftol=1e-16, gtol=1e-15,
                                                           time_budget=time_for_polish)
                if e_polish < best_e:
                    best_pts = pts_polish
                    best_e = e_polish
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
