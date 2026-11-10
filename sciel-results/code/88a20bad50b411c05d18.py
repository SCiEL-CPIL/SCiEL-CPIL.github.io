"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad_vec(pts):
    """Vectorized energy and gradient for log energy on sphere."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(d2, 1.0)  # avoid zero
    d = np.sqrt(d2)
    np.fill_diagonal(d, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, d2[i,j] = |x_i - x_j|^2
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _fibonacci_sphere(n):
    """Generate n points on unit sphere via golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _structured_grid(n):
    """Generate points on sphere using a structured grid approach."""
    # Use spherical coordinates with uniform area spacing
    n_theta = int(np.sqrt(n))
    n_phi = int(n / n_theta)
    if n_theta * n_phi < n:
        n_theta += 1
    theta_vals = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / n_theta
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    # Take first n points
    theta_flat = theta_flat[:n]
    phi_flat = phi_flat[:n]
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    pts = np.column_stack([x, y, z])
    return _project_to_sphere(pts)


def _riemannian_cg(pts, max_iter=2000, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient on the sphere."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    if deadline is not None and time.time() > deadline - 2:
        return pts
    
    e, g = _energy_and_grad_vec(pts)
    g_tan = _tangent_project(pts, g)
    
    # Initial search direction
    d = -g_tan
    
    # Riemannian CG
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Line search: find alpha that minimizes energy along geodesic
        # Geodesic: pts_new = pts * cos(alpha * |d_i|) + d_tan_i * sin(alpha * |d_i|) / |d_i|
        best_alpha = 1.0
        best_e = e
        
        # Backtracking line search
        alpha = 1.0
        for ls in range(30):
            # Compute new points along geodesic
            d_norms = np.linalg.norm(d, axis=1, keepdims=True)
            d_norms = np.maximum(d_norms, 1e-15)
            d_dir = d / d_norms
            cos_a = np.cos(alpha * d_norms)
            sin_a = np.sin(alpha * d_norms)
            pts_new = pts * cos_a + d_dir * sin_a
            pts_new = _project_to_sphere(pts_new)
            
            e_new, _ = _energy_and_grad_vec(pts_new)
            if e_new < best_e:
                best_e = e_new
                best_alpha = alpha
                pts = pts_new
            else:
                alpha *= 0.5
            if alpha < 1e-12:
                break
        
        if best_e >= e - 1e-14:
            # Try smaller step
            alpha = 0.01
            d_norms = np.linalg.norm(d, axis=1, keepdims=True)
            d_norms = np.maximum(d_norms, 1e-15)
            d_dir = d / d_norms
            cos_a = np.cos(alpha * d_norms)
            sin_a = np.sin(alpha * d_norms)
            pts_new = pts * cos_a + d_dir * sin_a
            pts_new = _project_to_sphere(pts_new)
            e_new, _ = _energy_and_grad_vec(pts_new)
            if e_new < e:
                pts = pts_new
                e = e_new
        
        e, g = _energy_and_grad_vec(pts)
        g_tan = _tangent_project(pts, g)
        
        # Check convergence
        if np.max(np.linalg.norm(g_tan, axis=1)) < tol:
            break
        
        # Riemannian CG update
        g_new_tan = _tangent_project(pts, g)
        beta = np.sum(g_new_tan * g_new_tan) / (np.sum(g_tan * g_tan) + 1e-30)
        d_new = -g_new_tan + beta * d
        # Parallel transport d to new point (approximate: project)
        d = _tangent_project(pts, d_new)
        g_tan = g_new_tan
    
    return pts


def _lbfgs_optimize(pts, deadline, max_iter=3000):
    """L-BFGS-B optimization with vectorized energy/grad."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad_vec(p)
        return e
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad_vec(p)
        g_tan = _tangent_project(p, g)
        return g_tan.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': max_iter * 10}
        )
        best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    except Exception:
        best_pts = pts
    
    return best_pts


def _polish(pts, deadline):
    """Final polish with Riemannian CG and tight tolerances."""
    if time.time() > deadline - 5:
        return pts
    
    polished = _riemannian_cg(pts, max_iter=500, tol=1e-12, deadline=deadline)
    return polished


def construct():
    deadline = time.time() + 115  # 115s safety margin within 120s limit
    
    # Generate multiple diverse starting points
    starts = []
    
    # 1. Fibonacci spiral (deterministic, good spread)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Structured grid
    starts.append(_structured_grid(N))
    
    # 3. Random starts with different seeds
    for seed in [42, 123, 777, 2024, 31415, 27182]:
        starts.append(_random_sphere(N, seed=seed))
    
    # 4. Perturbed Fibonacci
    rng = np.random.default_rng(999)
    fib = _fibonacci_sphere(N)
    for sigma in [0.1, 0.05, 0.2]:
        perturbed = fib + rng.normal(0, sigma, size=fib.shape)
        starts.append(_project_to_sphere(perturbed))
    
    # Optimize each start, track best
    best_pts = None
    best_energy = float('inf')
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # L-BFGS optimization
        result = _lbfgs_optimize(start, deadline, max_iter=2000)
        
        # Evaluate with precise energy
        e_result = _log_energy(result)
        
        if e_result < best_energy and e_result < float('inf'):
            best_energy = e_result
            best_pts = result
    
    # If we have time, do Riemannian CG polish on the best
    if best_pts is not None and time.time() < deadline - 5:
        polished = _polish(best_pts, deadline)
        e_polished = _log_energy(polished)
        if e_polished < best_energy and e_polished < float('inf'):
            best_pts = polished
            best_energy = e_polished
    
    # Final validation: ensure no points too close
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Verify no points closer than 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                mid = (best_pts[i] + best_pts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                best_pts[i] = best_pts[i] + 1e-8 * mid
                best_pts[j] = best_pts[j] - 1e-8 * mid
                best_pts[i] = best_pts[i] / np.linalg.norm(best_pts[i])
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts

# EVOLVE-BLOCK-END
