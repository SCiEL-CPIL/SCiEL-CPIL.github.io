import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _generate_test_points(num_points):
    """Generate random points on the unit sphere."""
    # Use a quasi-random sequence for better coverage
    i = np.arange(num_points) + 0.5
    z = 1.0 - 2.0 * i / num_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _objective(points_flat, test_points, beta):
    """
    Objective function: Soft-max approximation of the maximum angular distance
    from test points to the nearest configuration point.
    """
    points = points_flat.reshape(-1, 3)
    # Normalize points to ensure they are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Compute distances from test points to all configuration points
    # dot product: (num_test, N)
    dots = test_points @ points.T
    # Clip to [-1, 1] for numerical stability in arccos
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distances: (num_test, N)
    dists = np.arccos(dots)
    # Minimum distance for each test point: (num_test,)
    min_dists = np.min(dists, axis=1)
    
    # Soft-max approximation of the maximum
    # To avoid overflow, subtract the max
    max_d = np.max(min_dists)
    if max_d == 0:
        return 0.0
    # Use a stable log-sum-exp
    log_sum_exp = max_d + np.log(np.sum(np.exp(beta * (min_dists - max_d))))
    return log_sum_exp / beta

def _gradient(points_flat, test_points, beta):
    """
    Gradient of the objective function.
    """
    points = points_flat.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    dots = test_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    max_d = np.max(min_dists)
    if max_d == 0:
        return np.zeros_like(points_flat)
    
    # Weights for the soft-max
    weights = np.exp(beta * (min_dists - max_d))
    weights /= np.sum(weights)
    
    # For each test point, the nearest point contributes to the gradient
    # We need to find the index of the nearest point for each test point
    nearest_idx = np.argmin(dists, axis=1)
    
    grad_points = np.zeros_like(points)
    for i in range(len(test_points)):
        w = weights[i]
        j = nearest_idx[i]
        y = test_points[i]
        p = points[j]
        # Gradient of arccos(y.p) w.r.t p is -y / sqrt(1 - (y.p)^2)
        # But we are minimizing, so we want to move p to increase the distance?
        # No, we want to minimize the distance.
        # The gradient of the distance d = arccos(y.p) w.r.t p is -y / sin(d)
        # We want to decrease the max distance, so we move p in the direction of the gradient?
        # Wait, the objective is the max distance. We want to minimize it.
        # So we want to move p to reduce the distance to y.
        # The gradient of the distance w.r.t p points in the direction of increasing distance.
        # So we should move p in the negative gradient direction.
        # grad_d = -y / sin(d)
        # So the update for p is -grad_d * w = y / sin(d) * w
        # Let's check: if p is far from y, we want to move p towards y.
        # y / sin(d) is a vector in the direction of y (projected?).
        # Actually, the gradient of arccos(u) where u = y.p is -y / sqrt(1-u^2).
        # So the direction to increase distance is -y / sin(d).
        # The direction to decrease distance is y / sin(d).
        # So we add y / sin(d) * w to the gradient for p_j.
        
        d = dists[i, j]
        if np.sin(d) < 1e-6:
            continue
        grad_points[j] += w * y / np.sin(d)
        
    # The gradient of the soft-max is sum(w_i * grad(min_dist_i))
    # We computed grad_points as sum over i of w_i * (direction to decrease dist_i)
    # But we want the gradient of the objective, which is the max distance.
    # So the gradient of the objective w.r.t p_j is sum_{i: j is nearest} w_i * grad(dist_i)
    # grad(dist_i) = -y_i / sin(d_i)
    # So grad_obj = sum w_i * (-y_i / sin(d_i))
    # We computed grad_points[j] += w_i * y_i / sin(d_i)
    # So the actual gradient is -grad_points
    
    grad_points = -grad_points
    
    # Project gradient to the tangent space of the sphere
    # The constraint is ||p|| = 1. The tangent space is orthogonal to p.
    # grad_tangent = grad - (grad . p) p
    for j in range(N):
        p = points[j]
        g = grad_points[j]
        grad_points[j] = g - np.dot(g, p) * p
        
    return grad_points.flatten()

def construct():
    start_time = time.time()
    deadline = 145.0 # 150s limit, leave 5s margin
    
    # Generate test points
    test_points = _generate_test_points(2000)
    
    best_score = np.inf
    best_points = None
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    initial_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try a few restarts with different betas and initial perturbations
    restarts = 3
    betas = [10.0, 50.0, 100.0]
    
    for restart in range(restarts):
        if time.time() > deadline:
            break
            
        # Perturb the initial points slightly for different restarts
        if restart > 0:
            noise = np.random.normal(0, 0.1, initial_points.shape)
            current_points = initial_points + noise
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
        else:
            current_points = initial_points.copy()
            
        current_flat = current_points.flatten()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            res = minimize(
                _objective, 
                current_flat, 
                args=(test_points, beta),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                res_points = res.x.reshape(-1, 3)
                res_points /= np.linalg.norm(res_points, axis=1, keepdims=True)
                
                # Evaluate the true covering radius (max distance to nearest point)
                dots = test_points @ res_points.T
                dots = np.clip(dots, -1.0, 1.0)
                dists = np.arccos(dots)
                min_dists = np.min(dists, axis=1)
                score = np.max(min_dists)
                
                if score < best_score:
                    best_score = score
                    best_points = res_points
                    current_flat = res.x
            else:
                # If optimization failed, use the last point
                current_flat = res.x
        
        # Update current_flat for the next beta/restart
        current_points = current_flat.reshape(-1, 3)
        current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
        current_flat = current_points.flatten()

    if best_points is None:
        return initial_points
        
    return best_points
