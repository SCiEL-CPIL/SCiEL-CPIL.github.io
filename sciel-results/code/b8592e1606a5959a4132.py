"""Minimal logarithmic energy of 100 points on the unit sphere."""
import math
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START

_DEADLINE = None


def _deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140.0  # safety margin before 150s
    return _DEADLINE


def _time_left():
    return max(0.0, _deadline() - time.time())


def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _random_points(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    terms = []
    n = pts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_grad_fast(pts):
    """Compute energy and gradient vectorized. Returns (energy, grad)."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * (-1)
    # Wait, E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    # dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 ... let me re-derive
    # For pair (i,j): term = -log|x_i - x_j|
    # d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # But we need to be careful: for each pair (i,j), both x_i and x_j get contributions.
    
    # Using the diff matrix:
    # For point i, contribution from all j != i: -(x_i - x_j)/|x_i - x_j|^2
    # This equals (x_j - x_i)/|x_i - x_j|^2
    
    # dist_safe to avoid div by zero
    dist_safe = np.where(dist > 1e-15, dist, 1.0)
    inv_dist2 = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i,j]
    # = sum_j pts[j] * inv_dist2[i,j] - pts[i] * sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = np.sum(pts[np.newaxis, :, :] * inv_dist2[:, np.newaxis, :], axis=1) - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _riemannian_flow(pts, budget, initial_lr=0.05, patience=1000):
    """Gradient flow on the sphere with adaptive learning rate."""
    pts = _project_to_sphere(pts.copy())
    start = time.time()
    best_pts = pts.copy()
    best_e = _energy_fsum(pts)
    lr = initial_lr
    stale = 0
    
    while time.time() - start < budget and stale < patience:
        e, grad = _energy_grad_fast(pts)
        if not np.isfinite(e):
            break
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            stale = 0
        else:
            stale += 1
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adaptive step
        norm_g = np.linalg.norm(tangent_grad)
        if norm_g > 1e-12:
            step = lr * tangent_grad / norm_g
        else:
            break
        
        # Riemannian exponential map
        pts_new = _project_to_sphere(pts + step)
        
        # Check for improvement
        e_new, _ = _energy_grad_fast(pts_new)
        if np.isfinite(e_new) and e_new < e:
            pts = pts_new
            if stale > 50:
                lr *= 0.95
        else:
            lr *= 0.9
            if lr < 1e-10:
                break
    
    return best_pts


def _polish_lbfgs(pts, budget):
    """Polish using L-BFGS-B on flattened coordinates with projection."""
    from scipy.optimize import minimize
    
    best_pts = _project_to_sphere(pts.copy())
    best_e = _energy_fsum(best_pts)
    start = time.time()
    
    def objective_and_grad(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_grad_fast(p)
        # Pullback gradient through projection
        # For simplicity, use the gradient as-is (approximation)
        return e, g.flatten()
    
    # Try a few different initial steps
    for lr_factor in [1.0, 0.5, 0.2]:
        if time.time() - start > budget:
            break
        
        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxfun': 5000,
                    'maxls': 40
                }
            )
            if res.success or res.status in [1, 2, 3]:
                p_new = _project_to_sphere(res.x.reshape(N, 3))
                e_new = _energy_fsum(p_new)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = p_new
        except Exception:
            pass
    
    return best_pts


def _newton_polish(pts, max_iter=50, tol=1e-12):
    """Newton's method on the sphere for final high-precision refinement."""
    pts = _project_to_sphere(pts.copy())
    
    for _ in range(max_iter):
        e, grad = _energy_grad_fast(pts)
        if not np.isfinite(e):
            break
        
        # Project to tangent space
        tgrad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Approximate Hessian: for log energy, the Hessian in tangent space
        # is complex. Use a simple diagonal approximation or trust region.
        # For now, use a damped Newton step with line search
        norm_g = np.linalg.norm(tgrad)
        if norm_g < tol:
            break
        
        # Simple gradient descent step with Armijo line search
        step_dir = -tgrad / (norm_g + 1e-15)
        alpha = 0.1
        for _ in range(20):
            pts_trial = _project_to_sphere(pts + alpha * step_dir)
            e_trial, _ = _energy_grad_fast(pts_trial)
            if np.isfinite(e_trial) and e_trial < e:
                pts = pts_trial
                break
            alpha *= 0.5
        else:
            break
    
    return pts


def construct():
    _deadline()
    
    # Phase 1: Start with Fibonacci spiral and Riemannian flow
    pts = _fibonacci_spiral()
    budget1 = min(20.0, _time_left() * 0.3)
    refined = _riemannian_flow(pts, budget1, initial_lr=0.05, patience=500)
    e_ref = _energy_fsum(refined)
    
    best_pts = refined.copy()
    best_e = e_ref
    
    # Phase 2: Try random restarts
    np.random.seed(12345)
    restart_budget = min(60.0, _time_left() * 0.4)
    restart_start = time.time()
    
    for seed in range(1, 50):
        if time.time() - restart_start > restart_budget or _time_left() < 15.0:
            break
        
        rnd_pts = _random_points(seed)
        per_budget = min(3.0, _time_left() * 0.05)
        if per_budget < 0.5:
            break
        
        r_pts = _riemannian_flow(rnd_pts, per_budget, initial_lr=0.08, patience=200)
        e_r = _energy_fsum(r_pts)
        if e_r < best_e:
            best_e = e_r
            best_pts = r_pts.copy()
    
    # Phase 3: Polish the best result
    if _time_left() > 10.0:
        polish_budget = min(_time_left() * 0.5, 40.0)
        polished = _polish_lbfgs(best_pts, polish_budget)
        e_pol = _energy_fsum(polished)
        if e_pol < best_e:
            best_e = e_pol
            best_pts = polished.copy()
    
    # Phase 4: Final Newton polish for high precision
    if _time_left() > 3.0:
        newton_pts = _newton_polish(best_pts, max_iter=30, tol=1e-13)
        e_nt = _energy_fsum(newton_pts)
        if e_nt < best_e:
            best_e = e_nt
            best_pts = newton_pts.copy()
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                noise = rng.randn(3) * 1e-5
                best_pts[i] = _project_to_sphere(best_pts[i] + noise)
    
    return best_pts

# EVOLVE-BLOCK-END
