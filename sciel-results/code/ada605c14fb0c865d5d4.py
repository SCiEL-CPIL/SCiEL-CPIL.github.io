"""
Optimized approach for logarithmic energy minimization of 150 points on the sphere.
Strategy:
1. Hybrid approach: 
   a. Global structure search using a low-dimensional space-filling curve parameterization (7 params).
   b. Local refinement using full 450-dimensional L-BFGS on the sphere.
   c. Fallback to diverse multi-start 450-dimensional optimization.
2. Basin Hopping on the low-dimensional curve parameters to escape local minima of the curve shape.
3. High-precision Riemannian polish on the final 450-dimensional configuration.
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

def _generate_curve_points(params):
    """
    Generate 150 points on the sphere using a parameterized space-filling curve.
    params: [rx, ry, rz, phase, winding_mult, z_amp, z_freq]
    """
    rx, ry, rz, phase, winding_mult, z_amp, z_freq = params
    
    # Base Fibonacci indices
    i = np.arange(N) + 0.5
    
    # Z coordinate with sinusoidal perturbation
    z_base = 1.0 - 2.0 * i / N
    z = z_base + z_amp * np.sin(z_freq * i)
    # Clip z to [-1, 1] to ensure valid sphere
    z = np.clip(z, -1.0, 1.0)
    
    # Radius
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    
    # Azimuthal angle with winding multiplier
    golden_angle = np.pi * (1.0 + np.sqrt(5.0))
    phi = phase + winding_mult * golden_angle * i
    
    # Cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    pts = np.column_stack([x, y, z])
    
    # Apply rotation (Euler angles)
    # Rotation around Z
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pts = pts @ Rx.T
    
    # Rotation around Y
    cy, sy = np.cos(ry), np.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    pts = pts @ Ry.T
    
    # Rotation around X
    cx, sx = np.cos(rx), np.sin(rx)
    Rx2 = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    pts = pts @ Rx2.T
    
    # Final projection to sphere (should be close, but ensure)
    return _project_to_sphere(pts)

def _curve_energy_and_grad(params):
    """Energy and gradient with respect to curve parameters."""
    pts = _generate_curve_points(params)
    e, g_pts = _energy_and_gradient(pts)
    
    # Compute Jacobian of pts w.r.t params numerically (finite differences)
    # 7 params, so 7 forward differences is cheap
    grad_params = np.zeros(7)
    eps = 1e-6
    for k in range(7):
        params_plus = params.copy()
        params_plus[k] += eps
        pts_plus = _generate_curve_points(params_plus)
        e_plus = _energy_fast(pts_plus)
        grad_params[k] = (e_plus - e) / eps
        
    return e, grad_params

def _optimize_curve(params, deadline, maxiter=1000):
    """Optimize curve parameters using L-BFGS."""
    def fun_and_grad(p):
        e, g = _curve_energy_and_grad(p)
        return e, g

    try:
        res = minimize(
            fun_and_grad, params, method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        best_params = res.x
        best_e = _curve_energy_and_grad(res.x)[0]
    except Exception:
        best_params = params
        best_e = _curve_energy_and_grad(params)[0]
        
    return best_params, best_e

def _basin_hopping_curve(params, deadline, max_iters=20):
    """Basin hopping on curve parameters."""
    current_params = params.copy()
    current_energy = _curve_energy_and_grad(current_params)[0]
    best_params = current_params.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturb parameters
        # Scale perturbations based on parameter type
        scale = np.array([0.1, 0.1, 0.1, 0.5, 0.1, 0.05, 0.5])
        noise = rng.normal(0, 1, 7) * scale
        perturbed_params = current_params + noise
        
        # Optimize perturbed
        alloc = max(2.0, min(10.0, deadline - time.time() - 5.0))
        if alloc < 1.0:
            break
        opt_params, opt_e = _optimize_curve(perturbed_params, time.time() + alloc, maxiter=500)
        
        if opt_e < current_energy:
            current_params = opt_params
            current_energy = opt_e
            if current_energy < best_energy:
                best_params = current_params.copy()
                best_energy = current_energy
        else:
            # Sometimes accept worse to escape
            if rng.random() < 0.1:
                current_params = opt_params
                current_energy = opt_e

    return best_params, best_energy

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Curve-based optimization (Low dimensional)
    # Initial guess: Identity rotation, phase 0, winding 1, small amp/freq
    initial_params = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    # Optimize curve
    curve_deadline = deadline - 60.0 # Keep 60s for 450-var optimization
    if time.time() < curve_deadline:
        try:
            opt_params, opt_e = _optimize_curve(initial_params, curve_deadline, maxiter=500)
            # Basin hopping on curve
            bh_curve_deadline = min(deadline - 50.0, curve_deadline + 10.0)
            if time.time() < bh_curve_deadline:
                bh_params, bh_e = _basin_hopping_curve(opt_params, bh_curve_deadline, max_iters=15)
                if bh_e < opt_e:
                    opt_params, opt_e = bh_params, bh_e
            
            # Generate points from best curve params
            curve_pts = _generate_curve_points(opt_params)
            curve_e = _log_energy(curve_pts)
            
            if curve_e < best_energy:
                best_energy = curve_e
                best_pts = curve_pts.copy()
                
            # Polish curve points with 450-var L-BFGS
            if time.time() < deadline - 20.0:
                polish_deadline = deadline - 15.0
                polished_pts, polished_e = _optimize_on_sphere(curve_pts, polish_deadline, maxiter=2000)
                if polished_e < best_energy:
                    best_energy = polished_e
                    best_pts = polished_pts.copy()
        except Exception:
            pass

    # Phase 2: Original diverse multi-start optimization (Fallback / Comparison)
    start_configs = [
        _fibonacci_points(N),
        _sobol_points(N, seed=12345),
        _sobol_points(N, seed=99999),
        _random_points(N, seed=1),
        _jittered_fibonacci(N, jitter=0.1, seed=2),
        _jittered_fibonacci(N, jitter=0.2, seed=3),
    ]
    
    # Only run if we still have time and haven't found a very good solution
    # If curve found something < -2360, we might skip some starts
    for i, cfg in enumerate(start_configs):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        # Allocate less time if we already have a good result
        alloc = min(10.0, remaining - 20.0)
        if alloc < 3.0:
            break
        
        opt_pts, opt_e = _optimize_on_sphere(cfg, time.time() + alloc, maxiter=1000)
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_points(N)
        best_energy = _log_energy(best_pts)
    
    # Phase 3: Final Polish
    if time.time() < deadline - 10.0:
        p, e = _tangent_grad_descent_polish(best_pts, deadline - 5.0, max_iter=5000, base_lr=1e-3)
        if e < best_energy:
            best_energy = e
            best_pts = p
            
        # One more L-BFGS tight pass
        if time.time() < deadline - 5.0:
            p2, e2 = _optimize_on_sphere(best_pts, deadline - 2.0, maxiter=1000, ftol=1e-15, gtol=1e-13)
            if e2 < best_energy:
                best_energy = e2
                best_pts = p2

    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        best_pts = _project_to_sphere(_fibonacci_points(N))
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-9:
        best_pts = _project_to_sphere(_fibonacci_points(N))
    
    return best_pts
# EVOLVE-BLOCK-END
