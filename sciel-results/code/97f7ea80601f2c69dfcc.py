"""
Optimized logarithmic energy minimization for 150 points on the unit sphere.
Strategy:
1. Diverse initialization (Fibonacci, Icosahedron, Random).
2. Fast local minimization using Riemannian Conjugate Gradient (fixed and optimized).
3. Basin Hopping with adaptive perturbation to escape local minima.
4. High-precision polishing with L-BFGS.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
_DEADLINE = 145.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and tangential gradient for L-BFGS-B or Riemannian methods.
    x is flat array of shape (N*3,).
    Returns (energy, tangential_gradient_flat).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    # Compute pairwise squared distances
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi.xj = 2 - 2 xi.xj (since on sphere)
    # This is much faster than explicit diff and sum.
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    # Clamp small negative values due to floating point errors
    np.clip(dists_sq, 1e-15, None, out=dists_sq)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Using upper triangle for precision and to match definition
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(np.sqrt(dists_sq[iu]))
    energy = -np.sum(log_dists)
    
    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # Note: d_ij^2 = 2 - 2 x_i.x_j
    # Let W_ij = 1 / d_ij^2 for i != j, 0 otherwise.
    # grad_i = sum_j W_ij x_j - x_i sum_j W_ij
    
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    # Raw Euclidean gradient
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    # grad_tan = grad - (grad . p) p
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    return energy, grads_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_sphere(n, rng):
    """
    Generate points based on an icosahedron structure, subdivided and projected.
    This often provides a better symmetric start than Fibonacci.
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # We need n points. We can use a simple random jitter on an icosahedron-like grid
    # Or just use Fibonacci if icosahedron subdivision is too complex for arbitrary N.
    # For N=150, a Fibonacci sphere is a good baseline. 
    # Let's stick to Fibonacci + Random for robustness, as Icosahedron subdivision 
    # for non-matching N is tricky.
    return _fibonacci_sphere(n)

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _riemannian_cg(points, deadline, maxiter=1000, tol=1e-12):
    """
    Riemannian Conjugate Gradient method on the sphere.
    Fixed version: correct projection against current points.
    """
    pts = points.copy()
    n = len(pts)
    
    # Compute initial gradient
    _, grad_flat = _energy_and_grad(pts.flatten())
    grad = grad_flat.reshape(n, 3)
    
    # Project to tangent space
    dot_products = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot_products * pts
    
    # Initial direction
    d = -grad_tan
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return pts, _log_energy(pts)
    d = d / d_norm
    
    beta = 0.0
    alpha = 0.01  # Initial step size
    
    for it in range(maxiter):
        if time.time() > deadline - 1.0:
            break
        
        # Line search: backtracking to find sufficient decrease
        e_curr = _log_energy(pts)
        g_curr = grad_tan
        g_norm_sq = np.sum(g_curr * g_curr)
        
        if g_norm_sq < tol * tol:
            break
        
        # Armijo condition
        c1 = 1e-4
        rho = 0.5
        alpha_try = alpha
        improved = False
        
        for _ in range(20):
            new_pts = pts + alpha_try * d
            new_pts = _project_to_sphere(new_pts)
            e_new = _log_energy(new_pts)
            
            # Check Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * <g, d>
            if e_new <= e_curr + c1 * alpha_try * np.sum(g_curr * d):
                improved = True
                break
            alpha_try *= rho
        
        if not improved:
            # If no improvement found, try a smaller step
            alpha_try = max(alpha * 0.1, 1e-6)
            new_pts = pts + alpha_try * d
            new_pts = _project_to_sphere(new_pts)
            e_new = _log_energy(new_pts)
            
            if e_new >= e_curr:
                # No improvement possible, break
                break
        
        pts = new_pts
        
        # Compute new gradient
        _, grad_flat_new = _energy_and_grad(pts.flatten())
        grad_new = grad_flat_new.reshape(n, 3)
        
        # Project new gradient to tangent space of NEW points
        dot_products_new = np.sum(grad_new * pts, axis=1, keepdims=True)
        grad_tan_new = grad_new - dot_products_new * pts
        
        # Check convergence
        if np.linalg.norm(grad_tan_new) < tol:
            grad_tan = grad_tan_new
            break
        
        # Update direction (Polak-Ribiere)
        # beta = <g_new, g_new - g_old> / <g_old, g_old>
        num = np.sum(grad_tan_new * (grad_tan_new - grad_tan))
        den = np.sum(grad_tan * grad_tan)
        beta = num / max(den, 1e-15)
        beta = max(0.0, min(beta, 1.0))  # Clip to [0, 1] for stability
        
        d = -grad_tan_new + beta * d
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            d = -grad_tan_new
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                break
        d = d / d_norm
        
        # Update alpha estimate
        alpha = max(alpha_try, 1e-6)
        alpha = min(alpha, 1.0)
        
        grad_tan = grad_tan_new
    
    return pts, _log_energy(pts)

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    # Check if we have time
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        if time.time() > deadline - 1.0:
            return points, _log_energy(points)
            
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping_single(init_points, deadline, rng, max_iter=50, T_init=0.1):
    """
    Basin hopping with adaptive perturbation and Riemannian CG for local minimization.
    """
    current_points, current_energy = _riemannian_cg(init_points, deadline, maxiter=200, tol=1e-10)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95
    no_improve_count = 0
    max_no_improve = 15

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Adaptive perturbation
        if no_improve_count > 10:
            n_perturb = rng.randint(5, 15)
            step_scale = 2.0 + no_improve_count * 0.1
        elif no_improve_count > 5:
            n_perturb = rng.randint(3, 10)
            step_scale = 1.5
        else:
            n_perturb = rng.randint(2, 6)
            step_scale = 1.0

        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * step_scale
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Use Riemannian CG for local minimization
        if time.time() < deadline - 3.0:
            new_points, new_energy = _riemannian_cg(new_points, deadline, maxiter=300, tol=1e-10)
        else:
            new_points, new_energy = _local_minimize(new_points, deadline, maxiter=50)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        T *= cooling

    return best_points, best_energy

def _polish_high_precision(pts, max_time):
    """High-precision final polish using Riemannian CG then L-BFGS."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    # First pass: Riemannian CG for fast convergence
    if max_time > 2.0:
        cg_deadline = t0 + max_time * 0.6
        pts_cg, e_cg = _riemannian_cg(pts, cg_deadline, maxiter=1000, tol=1e-12)
        pts = pts_cg
    
    # Second pass: L-BFGS with very tight tolerances
    remaining = max_time - (time.time() - t0)
    if remaining < 0.5:
        return pts, _log_energy(pts)
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 10000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')

    # Phase 1: Diverse Initialization and Local Minimization
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points with different seeds
    for seed in range(3):
        r = np.random.RandomState(seed * 100 + 1)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci with varying noise levels
    for noise in [0.05, 0.1, 0.2]:
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * noise
        inits.append(_project_to_sphere(perturbed))

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Use Riemannian CG for faster local minimization
        x_res, e_res = _riemannian_cg(init, deadline, maxiter=500, tol=1e-10)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    if best_x is not None and time.time() < deadline - 15.0:
        x_bh, e_bh = _basin_hopping_single(best_x, deadline, rng, max_iter=60, T_init=0.08)
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh

    # Phase 3: Final High-Precision Polish
    if best_x is not None and time.time() < deadline - 3.0:
        polish_time = deadline - time.time() - 1.5
        if polish_time > 1.0:
            x_final, e_final = _polish_high_precision(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
