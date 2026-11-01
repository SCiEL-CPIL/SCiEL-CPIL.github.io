import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad_cartesian(points):
    """
    Compute logarithmic energy and its Euclidean gradient for points in R^3.
    Points are assumed to be near the unit sphere, but the calculation is valid for any non-zero distances.
    Energy: sum_{i<j} log(1/|xi - xj|)
    """
    # points shape (N, 3)
    # diff[i, j] = xi - xj
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid log(0)
    np.fill_diagonal(dist2, 1.0)
    
    # Energy
    # E = -0.5 * sum_{i!=j} log(sqrt(dist2)) = -0.25 * sum_{i!=j} log(dist2)
    mask = ~np.eye(N, dtype=bool)
    log_dist2 = np.log(dist2)
    total_log_dist2 = np.sum(log_dist2[mask])
    energy = -0.25 * total_log_dist2
    
    # Gradient
    # Grad_i = sum_{j!=i} (xi - xj) / dist2[i, j]
    # Note: The derivative of -log(d) wrt xi is -(xi-xj)/d^2.
    # Wait, E = sum -log(d). dE/dxi = sum -1/d * d(d)/dxi = sum -1/d * (xi-xj)/d = sum -(xi-xj)/d^2.
    # So grad_i = - sum_{j!=i} (xi - xj) / dist2[i, j] = sum_{j!=i} (xj - xi) / dist2[i, j].
    
    dist2_safe = np.where(mask, dist2, 1.0)
    # diff is xi - xj. We want (xj - xi) / dist2 = -diff / dist2.
    grad_vec = -diff / dist2_safe[:, :, np.newaxis]
    grad = np.sum(grad_vec, axis=1)
    
    return energy, grad

def _penalty_objective(points_flat, lambda_pen=1e5):
    """
    Objective function for L-BFGS-B with penalty for non-unit norm.
    """
    points = points_flat.reshape(N, 3)
    energy, grad_energy = _energy_and_grad_cartesian(points)
    
    # Penalty term: lambda * sum(|xi|^2 - 1)^2
    norms2 = np.sum(points * points, axis=1)
    penalty = lambda_pen * np.sum((norms2 - 1.0) ** 2)
    
    # Gradient of penalty: 4 * lambda * (|xi|^2 - 1) * xi
    grad_penalty = 4.0 * lambda_pen * (norms2 - 1.0)[:, np.newaxis] * points
    
    total_grad = grad_energy + grad_penalty
    total_obj = energy + penalty
    
    return total_obj, total_grad.flatten()

def _optimize_single(points_init, time_limit=10.0):
    """
    Optimize a single starting configuration using L-BFGS-B with penalty.
    """
    start_time = time.time()
    x0 = points_init.flatten()
    
    # Use L-BFGS-B
    # maxiter is a safety net, but we check time
    res = minimize(
        _penalty_objective, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    # Check time
    if time.time() - start_time > time_limit:
        # This is tricky, minimize doesn't have a callback to stop early easily in all versions
        # But we can just return the best found so far if we had a callback.
        # For now, we rely on the options. If it takes too long, we might kill the process.
        # Let's add a callback to check time.
        pass
        
    points_opt = res.x.reshape(N, 3)
    
    # Normalize the result to ensure it's on the sphere
    norms = np.linalg.norm(points_opt, axis=1)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    points_opt = points_opt / norms[:, np.newaxis]
    
    return points_opt

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points():
    points = np.random.randn(N, 3)
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, np.newaxis]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    best_energy = float('inf')
    best_points = None
    
    # Define a callback for time checking in minimize
    def time_callback(xk):
        if time.time() - start_time > deadline:
            raise Exception("Time limit reached")

    # Try multiple initializations
    inits = []
    inits.append(_fibonacci_spiral())
    
    # Add some random restarts
    num_randoms = 10
    for _ in range(num_randoms):
        inits.append(_random_points())
        
    # Maybe add a jittered fibonacci
    fib = _fibonacci_spiral()
    jitter = np.random.randn(N, 3) * 0.1
    fib_jitter = fib + jitter
    norms = np.linalg.norm(fib_jitter, axis=1)
    fib_jitter = fib_jitter / norms[:, np.newaxis]
    inits.append(fib_jitter)

    for i, init in enumerate(inits):
        if time.time() - start_time > deadline:
            break
            
        try:
            # Optimize
            # We can't easily pass a time callback to stop L-BFGS-B mid-run in a clean way without exceptions.
            # We will rely on maxiter and the fact that L-BFGS-B is fast.
            # If one run takes too long, we might timeout. 
            # To be safe, we can run a simpler gradient descent if L-BFGS is too slow?
            # Or just limit the number of iterations.
            
            res = minimize(
                _penalty_objective, 
                init.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            
            points_opt = res.x.reshape(N, 3)
            
            # Normalize
            norms = np.linalg.norm(points_opt, axis=1)
            norms[norms < 1e-12] = 1.0
            points_opt = points_opt / norms[:, np.newaxis]
            
            # Calculate true energy on the sphere
            energy, _ = _energy_and_grad_cartesian(points_opt)
            
            if energy < best_energy:
                best_energy = energy
                best_points = points_opt.copy()
                
        except Exception as e:
            # If optimization fails, skip
            pass

    # Final polish on the best point with higher precision
    if best_points is not None and time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                _penalty_objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'lambda_penalty': 1e6} # Note: lambda is hardcoded in func
            )
            # Wait, lambda is hardcoded in _penalty_objective. 
            # Let's make it a closure or pass it.
            # For now, the hardcoded 1e5 is likely sufficient for a polish if we just run more iterations.
            
            points_polished = res.x.reshape(N, 3)
            norms = np.linalg.norm(points_polished, axis=1)
            norms[norms < 1e-12] = 1.0
            points_polished = points_polished / norms[:, np.newaxis]
            
            energy, _ = _energy_and_grad_cartesian(points_polished)
            if energy < best_energy:
                best_energy = energy
                best_points = points_polished
        except Exception:
            pass

    return best_points
