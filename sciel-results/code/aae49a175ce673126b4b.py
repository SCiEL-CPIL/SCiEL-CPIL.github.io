"""Rewritten program using L-BFGS-B with manifold gradient handling and Basin Hopping."""
import numpy as np
import math
import time
from scipy.optimize import minimize, basinhopping

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere (or normalized inside).
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
        u: normalized points
    """
    # Ensure points are normalized for calculation
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = pts / norms
    
    # Pairwise differences
    diff = u[:, None, :] - u[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clip to avoid log(0) and division by 0
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy calculation
    # E = sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d_sq) * 0.5? No.
    # E = sum_{i<j} -log(d_ij) = -0.5 sum_{i<j} log(d_ij^2)
    # Let S = sum_{i!=j} log(d_sq_ij). Then sum_{i<j} log(d_sq_ij) = S/2.
    # E = -0.5 * (S/2) = -0.25 * S.
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient calculation
    # dE/du_i = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2
    # Note: The gradient of -log(||u_i - u_j||) w.r.t u_i is -(u_i - u_j)/||u_i - u_j||^2 = (u_j - u_i)/||u_i - u_j||^2
    # Since E = sum -log(d), dE = sum -(1/d) dd.
    # d(d_ij) = (u_i - u_j) . du_i / d_ij.
    # dE/d u_i = sum_{j!=i} - (u_i - u_j) / ||u_i - u_j||^2 = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2.
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # We want sum_j w_ij (u_j - u_i) where w_ij = 1/||u_i - u_j||^2
    # sum_j w_ij u_j - u_i sum_j w_ij
    # Matrix form: (W^T u)^T - u (W 1)^T
    # W is symmetric, so W^T = W.
    
    w_sum = np.sum(inv_dist_sq, axis=1)
    
    # term1[i] = sum_j inv_dist_sq[i,j] * u[j]
    term1 = inv_dist_sq @ u
    # term2[i] = u[i] * w_sum[i]
    term2 = u * w_sum[:, None]
    
    grad_raw = term1 - term2
    
    # Project to tangent space: grad_tan = grad_raw - (grad_raw . u_i) u_i
    dot = np.sum(grad_raw * u, axis=1)
    grad_tan = grad_raw - dot[:, None] * u
    
    return energy, grad_tan, u

def _objective_and_grad(x_flat):
    """
    Objective function and gradient for scipy.minimize.
    x_flat: (3N,) vector of unnormalized coordinates.
    """
    x = x_flat.reshape(N, 3)
    
    # Get energy and tangent gradient, and the normalized points u
    energy, grad_tan, u = _energy_and_tangent_grad(x)
    
    # Compute gradient in R^3 space for the normalized points
    # f(x) = E(x/||x||)
    # df/dx = (dE/du) . (du/dx)
    # du_i/dx_i = (1/||x_i||) I - (1/||x_i||^3) x_i x_i^T
    # grad_tan is orthogonal to u_i, so (grad_tan . u_i) = 0.
    # Thus, grad_tan . ( - x_i x_i^T / ||x_i||^3 ) = 0.
    # So grad_x_i = grad_tan_i / ||x_i||
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    grad_x = grad_tan / norms
    
    return energy, grad_x.flatten()

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_only(x_flat):
    """
    Scalar energy function for basin hopping.
    """
    x = x_flat.reshape(N, 3)
    energy, _, _ = _energy_and_tangent_grad(x)
    return energy

def _grad_only(x_flat):
    """
    Gradient-only function for scipy minimize.
    """
    _, g = _objective_and_grad(x_flat)
    return g

def _initial_antipodal():
    """Generate antipodal pairs on the sphere."""
    pts = []
    rng = np.random.default_rng(123)
    for _ in range(N // 2):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        pts.append(v)
        pts.append(-v)
    return np.array(pts)

def _optimize_lbfgs(start_pts, maxiter=2000, ftol=1e-10, gtol=1e-8):
    """
    Optimize using L-BFGS-B.
    """
    x0 = start_pts.flatten()
    
    options = {
        'maxiter': maxiter,
        'maxfun': maxiter * 5,
        'ftol': ftol,
        'gtol': gtol,
        'iprint': 0
    }
    
    res = minimize(
        _energy_only, 
        x0, 
        method='L-BFGS-B', 
        jac=_grad_only, 
        options=options
    )
    
    return res.x.reshape(N, 3)

def _optimize_lbfgs_tight(start_pts, maxiter=5000, ftol=1e-16, gtol=1e-14):
    """
    Ultra-tight polish pass for near-converged configurations.
    """
    x0 = start_pts.flatten()
    
    options = {
        'maxiter': maxiter,
        'maxfun': maxiter * 5,
        'ftol': ftol,
        'gtol': gtol,
        'iprint': 0
    }
    
    res = minimize(
        _energy_only, 
        x0, 
        method='L-BFGS-B', 
        jac=_grad_only, 
        options=options
    )
    
    return res.x.reshape(N, 3)

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci
    best_pts = _initial_fibonacci()
    best_energy, _, _ = _energy_and_tangent_grad(best_pts)
    
    # First, do a quick L-BFGS optimization from Fibonacci
    if time.time() < deadline - 60:
        try:
            opt_pts = _optimize_lbfgs(best_pts, maxiter=500, ftol=1e-10, gtol=1e-8)
            opt_pts = _project_to_sphere(opt_pts)
            e_opt, _, _ = _energy_and_tangent_grad(opt_pts)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = opt_pts
        except Exception:
            pass
    
    # Phase 1: Basin Hopping from the optimized Fibonacci start
    if time.time() < deadline - 30:
        try:
            def _basin_hopping_minimize(x):
                # Local minimization using L-BFGS-B
                res = minimize(
                    _energy_only, 
                    x, 
                    method='L-BFGS-B', 
                    jac=_grad_only, 
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'iprint': 0}
                )
                return res.x, res.fun, res.success
                
            x0 = best_pts.flatten()
            
            res_bh = basinhopping(
                _energy_only, 
                x0, 
                minimizer_kwargs={
                    'method': 'L-BFGS-B', 
                    'jac': _grad_only, 
                    'options': {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'iprint': 0}
                },
                stepsize=0.3,
                niter=30,
                T=50.0,
                disp=False
            )
            
            if res_bh.fun < best_energy:
                best_energy = res_bh.fun
                best_pts = res_bh.x.reshape(N, 3)
                
        except Exception:
            pass
    
    # Phase 2: High-precision polish on the best found
    if time.time() < deadline - 15:
        try:
            final_pts = _optimize_lbfgs(best_pts, maxiter=5000, ftol=1e-14, gtol=1e-12)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Additional polish if time remains
    if time.time() < deadline - 8:
        try:
            final_pts = _optimize_lbfgs(best_pts, maxiter=5000, ftol=1e-14, gtol=1e-12)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
        except Exception:
            pass

    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
