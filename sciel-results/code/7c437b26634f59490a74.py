"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_gradient(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Uses efficient O(N^2) computation.
    
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient of -log(d_ij) w.r.t. x_i: (x_i - x_j) / d_ij^2
    """
    n = len(pts)
    # Pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    
    # Pairwise distances
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Compute energy using upper triangle only
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    d_upper = np.maximum(d_upper, 1e-15)
    energy = float(math.fsum(-np.log(d_upper)))
    
    # Compute gradient
    # For each pair (i,j) with i<j:
    #   grad[i] += (x_i - x_j) / d_ij^2
    #   grad[j] += (x_j - x_i) / d_ij^2 = -(x_i - x_j) / d_ij^2
    # 
    # Using the full matrix approach:
    # grad[i] = sum_j (x_i - x_j) / d_ij^2  (for all j != i)
    
    dists_safe = np.where(mask, dists, 1.0)
    inv_d2 = np.where(mask, 1.0 / (dists_safe ** 2), 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_points():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(rng, n):
    """Generate n random points on unit sphere."""
    # Use uniform sampling on sphere
    z = rng.uniform(-1, 1, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(1 - z**2)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _tangent_project(grad, pts):
    """Project gradient to tangent space of the sphere at each point."""
    # grad_tangent = grad - (grad . x) * x
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - dot * pts

def _lgbfgs_optimize(pts, time_limit=30.0, maxiter=2000):
    """
    Optimize points on sphere using L-BFGS-B in Cartesian coordinates
    with tangent-space gradient projection.
    """
    start_time = time.time()
    pts = pts.copy()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        energy, grad = _energy_and_gradient(x)
        grad_tan = _tangent_project(grad, x)
        return energy, grad_tan.ravel()
    
    def objective(x_flat):
        return objective_and_grad(x_flat)[0]
    
    def jac(x_flat):
        return objective_and_grad(x_flat)[1]
    
    try:
        result = minimize(
            objective,
            pts.ravel(),
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
        )
        pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
        energy = _energy_and_gradient(pts_opt)[0]
        return pts_opt, energy
    except Exception as e:
        # If optimization fails, return original
        energy = _energy_and_gradient(pts)[0]
        return pts, energy

def _gradient_descent(pts, time_limit=20.0, lr_init=0.01):
    """
    Gradient descent with adaptive learning rate in tangent space.
    """
    start_time = time.time()
    pts = pts.copy()
    best_energy = _energy_and_gradient(pts)[0]
    best_pts = pts.copy()
    lr = lr_init
    
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_gradient(pts)
        grad_tan = _tangent_project(grad, pts)
        
        # Adaptive step size
        grad_norm = np.max(np.linalg.norm(grad_tan, axis=1))
        if grad_norm > 1e-10:
            lr = min(0.05, 0.01 / (grad_norm * 0.5 + 1))
        
        pts_new = pts - lr * grad_tan
        pts_new = _project_to_sphere(pts_new)
        
        energy_new = _energy_and_gradient(pts_new)[0]
        
        if energy_new < energy:
            pts = pts_new
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()
            lr *= 1.05  # slightly increase if progress
        else:
            lr *= 0.5
    
    return best_pts, best_energy

def _simulated_annealing(pts, time_limit=30.0, T_init=0.1, T_min=1e-6, cooling=0.995):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    pts = pts.copy()
    energy = _energy_and_gradient(pts)[0]
    best_pts = pts.copy()
    best_energy = energy
    T = T_init
    rng = np.random.default_rng(123)
    
    while time.time() - start_time < time_limit and T > T_min:
        # Random perturbation
        perturb = rng.normal(0, 0.05, size=pts.shape)
        # Project perturbation to tangent space
        perturb = _tangent_project(perturb, pts)
        
        pts_new = pts + perturb
        pts_new = _project_to_sphere(pts_new)
        
        energy_new = _energy_and_gradient(pts_new)[0]
        delta = energy_new - energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = pts_new
            energy = energy_new
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_energy

def _high_precision_polish(pts, time_limit=15.0):
    """
    High-precision final polish using very small steps.
    """
    start_time = time.time()
    pts = pts.copy()
    best_energy = _energy_and_gradient(pts)[0]
    best_pts = pts.copy()
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_gradient(pts)
        grad_tan = _tangent_project(grad, pts)
        
        grad_norm = np.max(np.linalg.norm(grad_tan, axis=1))
        if grad_norm < 1e-15:
            break
        
        # Very small step
        step = min(1e-4, 1e-5 / (grad_norm * 10 + 1))
        
        pts_new = pts - step * grad_tan
        pts_new = _project_to_sphere(pts_new)
        
        energy_new = _energy_and_gradient(pts_new)[0]
        
        if energy_new < energy:
            pts = pts_new
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    def update_best(pts, energy):
        nonlocal best_energy, best_pts
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Strategy 1: Fibonacci spiral + L-BFGS-B (primary)
    if time.time() < deadline - 80.0:
        pts = _fibonacci_points()
        pts_opt, energy = _lgbfgs_optimize(pts, time_limit=40.0, maxiter=3000)
        update_best(pts_opt, energy)
    
    # Strategy 2: L-BFGS-B from slightly perturbed Fibonacci
    if time.time() < deadline - 60.0:
        rng = np.random.default_rng(42)
        pts = _fibonacci_points() + rng.normal(0, 0.01, size=(N, 3))
        pts = _project_to_sphere(pts)
        pts_opt, energy = _lgbfgs_optimize(pts, time_limit=25.0, maxiter=2000)
        update_best(pts_opt, energy)
    
    # Strategy 3: Simulated annealing from best so far
    if time.time() < deadline - 40.0 and best_pts is not None:
        pts_sa, energy_sa = _simulated_annealing(best_pts, time_limit=30.0)
        update_best(pts_sa, energy_sa)
        
        # Follow up with L-BFGS-B after SA
        if time.time() < deadline - 20.0:
            pts_opt, energy = _lgbfgs_optimize(pts_sa, time_limit=15.0, maxiter=1000)
            update_best(pts_opt, energy)
    
    # Strategy 4: Random restarts with L-BFGS-B
    rng = np.random.default_rng(1234)
    for restart in range(5):
        if time.time() > deadline - 25.0:
            break
        pts = _random_points(rng, N)
        # Quick gradient descent first
        pts_gd, energy_gd = _gradient_descent(pts, time_limit=5.0)
        # Then L-BFGS-B
        pts_opt, energy = _lgbfgs_optimize(pts_gd, time_limit=8.0, maxiter=500)
        update_best(pts_opt, energy)
    
    # Strategy 5: More SA iterations if time permits
    if time.time() < deadline - 20.0 and best_pts is not None:
        rng2 = np.random.default_rng(5678)
        pts_sa2, energy_sa2 = _simulated_annealing(best_pts, time_limit=15.0, T_init=0.05)
        update_best(pts_sa2, energy_sa2)
    
    # Final high-precision polish
    if time.time() < deadline - 5.0 and best_pts is not None:
        pts_polish, energy_polish = _high_precision_polish(best_pts, time_limit=12.0)
        update_best(pts_polish, energy_polish)
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points()
    
    # Final normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close together and fix
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                rng = np.random.default_rng(i * 1000 + j)
                best_pts[j] += 1e-7 * rng.normal(size=3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
