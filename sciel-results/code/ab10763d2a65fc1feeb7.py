import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_grad_flat(flat):
    """
    Compute logarithmic energy and gradient from flattened point array.
    flat: (3*N,) array of point coordinates.
    Returns: (energy, grad_flat)
    """
    pts = flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # = -0.5 * sum_{i!=j} log(d_ij)
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # So energy = -0.25 * sum_{i!=j} log(dist_sq_ij)
    dsq_masked = np.where(mask, dist_sq, 1.0)
    energy = -0.25 * np.sum(np.log(dsq_masked))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2 * (-1) * 2 * ... 
    # Actually: E = -sum_{i<j} log(d_ij), dE/dx_i = sum_{j!=i} (x_i - x_j)/d_ij^2
    # But we need to account for the projection constraint.
    # For now, compute the unconstrained gradient and let L-BFGS handle it.
    inv_dsq = np.where(mask, 1.0 / np.maximum(dist_sq, 1e-12), 0.0)
    grad = np.sum(diff * inv_dsq[:, :, np.newaxis], axis=1)
    
    # Project gradient onto tangent space
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return energy, grad.ravel()

def _optimize_single(points, time_limit=10.0):
    """Optimize a single starting configuration using L-BFGS-B."""
    flat = points.ravel()
    
    result = minimize(
        _energy_grad_flat,
        flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    opt_pts = result.x.reshape(N, 3)
    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    opt_pts = opt_pts / norms
    
    return opt_pts, result.fun

def _simulated_annealing_refine(points, time_limit=5.0):
    """Refine using simulated annealing to escape local minima."""
    current = points.copy()
    current_energy, _ = _energy_grad_flat(current.ravel())
    best = current.copy()
    best_energy = current_energy
    
    start_time = time.time()
    T = 0.1
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Random perturbation
        noise = np.random.randn(N, 3) * 0.01
        candidate = current + noise
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        candidate = candidate / norms
        
        cand_energy, _ = _energy_grad_flat(candidate.ravel())
        delta = cand_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current = candidate
            current_energy = cand_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        T *= cooling
    
    return best, best_energy

def construct():
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for evaluation overhead
    
    best_energy = float('inf')
    best_points = None
    
    # Try multiple starting configurations
    starts = []
    
    # Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    
    # Random starts
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    
    # Optimize each start
    for i, start_pts in enumerate(starts):
        if time.time() - start_time > timeout - 15.0:
            break
        
        # Use L-BFGS for optimization
        opt_pts, opt_energy = _optimize_single(start_pts, time_limit=15.0)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_pts.copy()
    
    # If we have time, do simulated annealing refinement on the best
    if best_points is not None and time.time() - start_time < timeout - 5.0:
        refined, refined_energy = _simulated_annealing_refine(
            best_points, 
            time_limit=min(10.0, timeout - (time.time() - start_time) - 2.0)
        )
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_points = refined
    
    # Final validation
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points
