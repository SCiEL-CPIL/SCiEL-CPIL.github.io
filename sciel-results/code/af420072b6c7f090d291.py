"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise squared distances efficiently:
    # |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2*x_i.x_j
    # Since pts are on unit sphere, |x_i|^2 = 1
    # So dist_sq[i,j] = 2 - 2*pts[i].pts[j]
    gram = pts @ pts.T  # (n, n)
    dist_sq = 2.0 - 2.0 * gram
    # Clamp small negative values from FP error
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    np.fill_diagonal(dist_sq, 1.0)

    # Energy: sum over i < j of -log(dist) = -0.5 * sum log(dist_sq) for i<j
    # sum_{i<j} log(dist_ij) = 0.5 * sum_{i != j} log(dist_ij)
    # So E = -0.5 * sum_{i != j} log(dist_ij) = -sum_{i != j} 0.5 * log(dist_ij)
    # More directly: E = -sum_{i<j} log(dist_ij)
    # Using: sum_{i<j} f(i,j) = 0.5 * (sum_{i,j} f(i,j) - sum_i f(i,i))
    log_dist = np.log(dist_sq)  # log(dist_sq) = 2*log(dist)
    # sum_{i<j} log(dist_ij) = 0.5 * (sum_{i,j} log(dist_sq_ij) - sum_i log(dist_sq_ii))
    # dist_sq_ii = 1, so log(dist_sq_ii) = 0
    sum_log_dist_sq = np.sum(log_dist)  # includes diagonal which is log(1)=0
    sum_log_dist = 0.5 * sum_log_dist_sq  # sum over i<j of log(dist_ij)
    energy = -sum_log_dist

    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_j (x_i - x_j) / dist_sq[i,j]  (with dist_sq[i,i] ignored)
    # = -x_i * sum_j 1/dist_sq[i,j] + sum_j x_j / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # sum_j 1/dist_sq[i,j] for each i
    row_sums = np.sum(inv_dist_sq, axis=1)  # (n,)
    # sum_j x_j / dist_sq[i,j] for each i
    # This is (inv_dist_sq @ pts) where inv_dist_sq[i,j] * pts[j]
    weighted_sum = inv_dist_sq @ pts  # (n, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum

    return energy, grad

def _project_to_tangent(pts, grad):
    """
    Project the Euclidean gradient to the tangent space of the unit sphere.
    T_p S^2 = { v in R^3 | v . p = 0 }
    Projection: g_tan = g - (g . p) p
    """
    # pts: (n, 3), grad: (n, 3)
    # Dot product: (n,)
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    return grad_tan

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _make_fibonacci_sphere(n):
    """Generate points on the sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z)) # Clamp to avoid sqrt of negative due to FP
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    # Generate random vectors and normalize
    pts = rng.normal(size=(n, 3))
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # List of initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_make_fibonacci_sphere(N))
    
    # 2. Random points (several seeds)
    for seed in [1, 2, 3, 42, 123, 777, 2024]:
        initializations.append(_random_sphere(N, seed))
        
    # 3. Perturbed Fibonacci
    rng = np.random.default_rng(99)
    fib = _make_fibonacci_sphere(N)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        perturbed = fib + rng.normal(0, scale, size=fib.shape)
        initializations.append(_normalize(perturbed))

    # Optimization function wrapper
    def obj_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        g_tan = _project_to_tangent(pts, g)
        return e, g_tan.ravel()

    # Run optimization for each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
            
        current_energy, _ = _energy_and_grad(init_pts)
        
        # If this initialization is already worse than the best found, skip if we have enough restarts
        if current_energy > best_energy and i > 3:
             # Allow a small tolerance for exploration
             if current_energy - best_energy > 3.0:
                 continue

        x0 = init_pts.ravel()
        
        try:
            # L-BFGS-B with tight tolerances
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1500,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            # Re-normalize the result
            res_pts = _normalize(result.x.reshape(N, 3))
            
            # Calculate final energy accurately
            res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
                
        except Exception as e:
            # If optimization fails, keep the initial points if they are better
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = init_pts.copy()
                
    # If we still have time, do a final polish on the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            x0 = best_pts.ravel()
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'disp': False
                }
            )
            res_pts = _normalize(result.x.reshape(N, 3))
            res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except:
            pass

    # High-precision steepest-descent polish with line search
    if best_pts is not None and time.time() < deadline - 3.0:
        pts = best_pts.copy()
        cur_e, _ = _energy_and_grad(pts)
        polish_deadline = deadline - 3.0
        
        # Steepest descent with backtracking line search in tangent space
        for _sd_iter in range(500):
            if time.time() > polish_deadline:
                break
            
            e, g = _energy_and_grad(pts)
            g_tan = _project_to_tangent(pts, g)
            
            # Check convergence
            max_grad = np.max(np.linalg.norm(g_tan, axis=1))
            if max_grad < 1e-10:
                break
            
            # Normalize gradient direction for stable step sizes
            grad_norm = np.linalg.norm(g_tan)
            if grad_norm < 1e-15:
                break
            direction = -g_tan / grad_norm
            
            # Backtracking line search
            step = 0.1
            improved = False
            for _ls_iter in range(20):
                # Move in tangent direction and normalize
                new_pts = pts + step * direction
                new_pts = _normalize(new_pts)
                new_e, _ = _energy_and_grad(new_pts)
                
                if new_e < cur_e - 1e-15:
                    pts = new_pts
                    cur_e = new_e
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                break
        
        res_energy, _ = _energy_and_grad(pts)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = pts.copy()

    # Final validity checks
    if best_pts is None:
        best_pts = _make_fibonacci_sphere(N)
        
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _make_fibonacci_sphere(N)
        
    if not np.all(np.isfinite(best_pts)):
        best_pts = _make_fibonacci_sphere(N)

    return best_pts

# EVOLVE-BLOCK-END
