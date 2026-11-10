"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _fibonacci_spiral(n, offset=0.0):
    """Generate Fibonacci spiral points on unit sphere with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # Sum log(1/d) = -sum log(d) over upper triangle
    upper = np.triu_indices(n, k=1)
    d = dist[upper]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return -math.fsum(math.log(d_i) for d_i in d)

def _log_energy_and_grad(points):
    """Compute logarithmic energy and its gradient on the sphere.
    
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    Then project to tangent space: grad_tangent = grad - (grad . x_i) * x_i
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: sum_j (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad[i] = sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: remove radial component
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - radial * points
    
    # Compute energy
    mask = ~np.eye(n, dtype=bool)
    upper = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[upper])
    d = np.maximum(d, 1e-15)
    energy = -math.fsum(math.log(d_i) for d_i in d)
    
    return energy, grad_tangent

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _gradient_descent(points, max_iter=2000, tol=1e-12, lr_init=0.01, lr_decay=0.999):
    """Gradient descent with projection to sphere."""
    points = _project_to_sphere(points.copy())
    energy, grad = _log_energy_and_grad(points)
    best_energy = energy
    best_points = points.copy()
    
    lr = lr_init
    for it in range(max_iter):
        # Take gradient step
        points_new = points - lr * grad
        points_new = _project_to_sphere(points_new)
        
        new_energy, new_grad = _log_energy_and_grad(points_new)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = points_new.copy()
        
        # Check convergence
        if abs(energy - new_energy) < tol:
            break
        
        energy = new_energy
        grad = new_grad
        points = points_new
        lr *= lr_decay
    
    return best_points, best_energy

def _lbfgs_optimize(points, max_iter=5000, tol=1e-12):
    """Use scipy L-BFGS-B for optimization."""
    from scipy.optimize import minimize
    
    points = _project_to_sphere(points.copy())
    
    def objective_and_grad(flat_points):
        pts = flat_points.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        return e, g
    
    result = minimize(
        objective_and_grad,
        points.ravel(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
    )
    
    opt_points = _project_to_sphere(result.x.reshape(-1, 3))
    opt_energy, _ = _log_energy_and_grad(opt_points)
    
    return opt_points, opt_energy

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Generate multiple initializations
    initializations = []
    
    # Base Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # Perturbed Fibonacci spirals with different offsets
    for offset in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Random initialization
    rng = np.random.default_rng(42)
    for _ in range(3):
        random_pts = rng.standard_normal((N, 3))
        initializations.append(_project_to_sphere(random_pts))
    
    # Try gradient descent on each initialization
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        try:
            gd_points, gd_energy = _gradient_descent(init_pts, max_iter=3000, tol=1e-13, lr_init=0.005, lr_decay=0.9995)
            
            if gd_energy < best_energy:
                best_energy = gd_energy
                best_points = gd_points.copy()
        except Exception:
            continue
    
    # Polish the best result with L-BFGS
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline:
        try:
            lbfgs_points, lbfgs_energy = _lbfgs_optimize(best_points, max_iter=10000, tol=1e-14)
            
            if lbfgs_energy < best_energy:
                best_energy = lbfgs_energy
                best_points = lbfgs_points.copy()
        except Exception:
            pass
    
    # Final polish: more iterations of gradient descent with very small learning rate
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline:
        try:
            final_points, final_energy = _gradient_descent(best_points, max_iter=5000, tol=1e-14, lr_init=0.001, lr_decay=0.9999)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = final_points.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    best_points = _project_to_sphere(best_points)
    
    # Validate: no points closer than 1e-9
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
        best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
