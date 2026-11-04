"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import qmc

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
    Energy and tangential gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    Returns (energy, tangential_gradient_flat).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # Using matrix operations for efficiency:
    # weights[i,j] = 1/d_ij^2 for i!=j, 0 otherwise
    # grads[i] = sum_j weights[i,j] x_j - x_i * sum_j weights[i,j]
    
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    # Raw Euclidean gradient
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy for the optimizer
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))

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

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _halton_sphere(n, rng):
    """Generate points using Halton sequence projected to sphere."""
    # Use scipy's Halton sampler
    sampler = qmc.Halton(d=3, scramble=True, seed=int(rng.randint(1, 10000)))
    points = sampler.random(n)
    # Map [0,1]^3 to sphere
    # One way: normalize, but this clusters at corners.
    # Better: use spherical coordinates from Halton
    # u, v in [0,1]
    # theta = 2*pi*u
    # z = 2*v - 1
    # r = sqrt(1-z^2)
    u = points[:, 0]
    v = points[:, 1]
    theta = 2 * np.pi * u
    z = 2 * v - 1
    r = np.sqrt(np.maximum(0, 1 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y, z])
    # Add small random perturbation to break symmetry
    pts += rng.randn(n, 3) * 0.01
    return _project_to_sphere(pts)

def _local_minimize(points, deadline, maxiter=1000, ftol=1e-12, gtol=1e-10):
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
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if time.time() > deadline - 1.0:
            return points, _log_energy(points)
            
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _riemannian_gradient_descent(points, deadline, max_iter=5000, tol=1e-14):
    """
    Riemannian Gradient Descent with Barzilai-Borwein step size.
    """
    pts = _project_to_sphere(points.copy())
    n = len(pts)
    
    # Compute initial gradient
    _, grad_flat = _energy_and_grad(pts.flatten())
    grad = grad_flat.reshape(n, 3)
    
    # Project gradient to tangent space
    dot_products = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot_products * pts
    
    # Initial step size
    alpha = 0.01
    
    prev_grad_tan = grad_tan.copy()
    prev_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 0.5:
            break
            
        # Compute BB step size
        dx = pts - prev_pts
        dg = grad_tan - prev_grad_tan
        
        num = np.sum(dx * dg)
        den = np.sum(dg * dg)
        
        if den > 1e-16:
            alpha = abs(num / den)
            alpha = np.clip(alpha, 1e-6, 1.0)
        
        # Gradient step
        new_pts = pts - alpha * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        # Check convergence
        if np.linalg.norm(new_pts - pts) < tol:
            pts = new_pts
            break
        
        # Update
        prev_pts = pts
        prev_grad_tan = grad_tan
        pts = new_pts
        
        # Compute new gradient
        _, grad_flat = _energy_and_grad(pts.flatten())
        grad = grad_flat.reshape(n, 3)
        dot_products = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - dot_products * pts
        
        # Early stop if gradient is small
        if np.linalg.norm(grad_tan) < 1e-12:
            break

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
    
    # 2. Random uniform points
    for seed in range(5):
        r = np.random.RandomState(seed * 137 + 7)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Halton sequence
    for seed in range(5):
        r = np.random.RandomState(seed * 53 + 11)
        inits.append(_halton_sphere(N, r))

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # First pass: L-BFGS-B
        x_res, e_res = _local_minimize(init, deadline, maxiter=200, ftol=1e-10, gtol=1e-8)
        
        # Second pass: Riemannian GD for refinement
        x_res, e_res = _riemannian_gradient_descent(x_res, deadline, max_iter=1000)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    if best_x is not None and time.time() < deadline - 15.0:
        current_points = best_x.copy()
        current_energy = best_e
        T = 0.1
        for it in range(50):
            if time.time() > deadline - 5.0:
                break
            # Perturb
            n_perturb = rng.randint(5, 15)
            idxs = rng.choice(N, n_perturb, replace=False)
            new_points = current_points.copy()
            for idx in idxs:
                rand_vec = rng.randn(3)
                tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
                norm = np.linalg.norm(tangent)
                if norm > 1e-15:
                    tangent /= norm
                    step = T * rng.uniform(0.5, 1.5)
                    new_point = new_points[idx] + step * tangent
                    new_points[idx] = new_point / np.linalg.norm(new_point)
            
            # Minimize
            x_res, e_res = _local_minimize(new_points, deadline, maxiter=100, ftol=1e-10, gtol=1e-8)
            x_res, e_res = _riemannian_gradient_descent(x_res, deadline, max_iter=500)
            
            if e_res < current_energy:
                current_points = x_res
                current_energy = e_res
                if current_energy < best_e:
                    best_e = current_energy
                    best_x = current_points.copy()
            T *= 0.9

    # Phase 3: Final High-Precision Polish
    if best_x is not None and time.time() < deadline - 2.0:
        # Use L-BFGS-B with very tight tolerances
        x_final, e_final = _local_minimize(best_x, deadline, maxiter=5000, ftol=1e-15, gtol=1e-14)
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
# EVOLVE-BLOCK-END
