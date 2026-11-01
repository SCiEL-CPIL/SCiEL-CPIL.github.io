import time
import math
import numpy as np
from scipy.spatial import cKDTree

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _energy_grad(points):
    """
    Compute logarithmic energy and its gradient.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: dE/dx_i = -sum_{j!=i} 1/|x_i - x_j| * (x_i - x_j) / |x_i - x_j|^2
             = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^3
    Note: The gradient of log(1/d) is -1/d * d(d)/dx.
    d(d)/dx_i = (x_i - x_j)/d. So dE/dx_i = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2.
    Wait, let's re-derive carefully.
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = sum_{j!=i} - (1/||x_i - x_j||) * d(||x_i - x_j||)/dx_i
    d(||x_i - x_j||)/dx_i = (x_i - x_j) / ||x_i - x_j||
    So dE/dx_i = sum_{j!=i} - (1/||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
               = sum_{j!=i} - (x_i - x_j) / ||x_i - x_j||^2
               = sum_{j!=i} (x_j - x_i) / ||x_i - x_j||^2
    """
    n = len(points)
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # (i, j) -> x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero on diagonal
    
    dist = np.sqrt(dist_sq)
    
    # Gradient: sum_{j!=i} (x_j - x_i) / dist^2
    # diff[i,j] = x_i - x_j
    # We want sum (x_j - x_i) / dist^2 = sum -diff[i,j] / dist^2
    # So grad[i] = - sum_{j!=i} diff[i,j] / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j inv_dist_sq[i,j] * (x_j - x_i)
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum_{i<j} -log(dist[i,j])
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_exp(p, v):
    """
    Exponential map on the unit sphere.
    p: point on sphere
    v: tangent vector at p
    Returns: exp_p(v)
    """
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return p
    return p * np.cos(theta) + (v / theta) * np.sin(theta)

def _optimize_riemannian(points, time_budget, max_iter=10000):
    """
    Riemannian gradient descent with momentum and adaptive step size.
    """
    start_time = time.time()
    p = points.copy()
    v = np.zeros_like(p) # Momentum
    best_energy = _energy_grad(p)[0]
    best_p = p.copy()
    
    # Adaptive step size parameters
    alpha = 0.01
    alpha_min = 1e-8
    alpha_max = 0.5
    momentum = 0.9
    
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        e, g = _energy_grad(p)
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        
        # Update momentum
        v = momentum * v + alpha * g_tan
        
        # Compute step direction
        # We use the exponential map for stability
        # Step size is effectively controlled by alpha and momentum
        # Let's scale v to ensure it's not too large
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        # Limit the step size to avoid instability
        max_step = 0.1
        scale = np.where(v_norm > max_step, max_step / v_norm, 1.0)
        v_scaled = v * scale
        
        # Apply exponential map
        p_new = _riemannian_exp(p, -v_scaled) # Gradient descent: move opposite to gradient
        
        # Check energy change
        e_new, _ = _energy_grad(p_new)
        
        if e_new < best_energy:
            best_energy = e_new
            best_p = p_new.copy()
            
        # Adaptive step size
        if e_new < e:
            # Successful step, can increase alpha slightly
            alpha = min(alpha * 1.05, alpha_max)
            p = p_new
        else:
            # Failed step, decrease alpha
            alpha = max(alpha * 0.5, alpha_min)
            if alpha < alpha_min * 1.1:
                # If step size is very small, try a random kick to escape
                if np.random.random() < 0.1:
                    p = p + np.random.randn(*p.shape) * 0.01
                    p = _project_to_sphere(p)
                    v = np.zeros_like(p)
                else:
                    break
            else:
                # Try a smaller step in the same direction
                p = p_new # Keep the new position even if energy increased? 
                          # Usually, we revert. Let's revert to be safe.
                p = p # Revert
                v = 0.5 * v # Dampen momentum

    return best_p

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)

def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)

def _golden_spiral(n):
    """Generate initial points using a golden spiral variant."""
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = 2.399963229728653 * i # Golden angle
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)

def construct():
    start_time = time.time()
    
    # List of initializations
    initializations = [
        ("fib", _fibonacci_spiral(N)),
        ("rand1", _random_points(N, seed=1)),
        ("rand2", _random_points(N, seed=2)),
        ("golden", _golden_spiral(N)),
    ]
    
    best_points = initializations[0][1].copy()
    best_energy = _energy_grad(best_points)[0]
    
    # Phase 1: Optimize each initialization for a short time
    time_per_init = min(10.0, (TIME_LIMIT * 0.5) / len(initializations))
    
    for name, init_points in initializations:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.5:
            break
        
        budget = min(time_per_init, TIME_LIMIT * 0.5 - elapsed)
        if budget < 1.0:
            continue
            
        optimized = _optimize_riemannian(init_points, budget)
        e = _energy_grad(optimized)[0]
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Iterative refinement with kicks
    # Use remaining time to refine the best solution, occasionally kicking it
    np.random.seed(42)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.9:
            break
        
        # Determine budget for this iteration
        remaining = TIME_LIMIT * 0.9 - elapsed
        if remaining < 2.0:
            break
            
        # Occasionally kick the best solution
        if np.random.random() < 0.3:
            kick_scale = 0.05 / (1 + elapsed / 10.0)
            kicked = best_points + np.random.randn(N, 3) * kick_scale
            kicked = _project_to_sphere(kicked)
            start_points = kicked
        else:
            start_points = best_points
            
        budget = min(remaining * 0.5, 15.0)
        
        optimized = _optimize_riemannian(start_points, budget)
        e = _energy_grad(optimized)[0]
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.98:
        remaining = TIME_LIMIT * 0.98 - elapsed
        if remaining > 1.0:
            final_opt = _optimize_riemannian(best_points, remaining * 0.9)
            e = _energy_grad(final_opt)[0]
            if e < best_energy:
                best_energy = e
                best_points = final_opt
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy_grad(best_points)[0]
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
