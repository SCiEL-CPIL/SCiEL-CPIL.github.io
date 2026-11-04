import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius (in radians) of the point configuration."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi
    
    if len(simplices) == 0:
        return np.pi
    
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        return np.pi
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius: angle between circumcenter direction and any vertex
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return np.max(radii)

def _smooth_objective(params, beta=50.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    params: flattened array of shape (3*N,)
    Returns: scalar objective value and gradient
    """
    points = params.reshape(N, 3)
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.zeros_like(params)
    
    if len(simplices) == 0:
        return np.pi, np.zeros_like(params)
    
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        return np.pi, np.zeros_like(params)
        
    u = n / norms[:, np.newaxis]
    
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    if max_r < 1e-10:
        return max_r, np.zeros_like(params)
    
    # Soft max: (1/beta) * log(sum(exp(beta * radii)))
    shifted = beta * (radii - max_r)
    sum_exp = np.sum(np.exp(shifted))
    smooth_val = max_r + np.log(sum_exp) / beta
    
    # Gradient of smooth objective w.r.t. points
    # d(smooth_val)/d(radii_i) = exp(beta * radii_i) / sum_exp
    weights = np.exp(shifted) / sum_exp
    
    # d(radii_i)/d(points) for each facet
    # radii_i = arccos(sum(u_i * a_i))
    # d(radii_i)/d(a_i) = -u_i / sqrt(1 - cos^2)
    # But we need to be careful with the chain rule through normalization
    
    grad_points = np.zeros((N, 3))
    
    for i in range(len(simplices)):
        if weights[i] < 1e-12:
            continue
        
        idx = simplices[i]
        i_idx, j_idx, k_idx = idx
        center = u[i]
        cos_t = cos_theta[i]
        
        # Derivative of arccos(x) is -1/sqrt(1-x^2)
        denom = np.sqrt(np.clip(1.0 - cos_t * cos_t, 1e-12, None))
        factor = -weights[i] / denom
        
        # d(cos_theta)/d(point) = center for the vertex
        grad_points[i_idx] += factor * center
        grad_points[j_idx] += factor * center
        grad_points[k_idx] += factor * center
    
    # Account for normalization constraint
    # If points are on unit sphere, we project gradient to tangent space
    # grad_tangent = grad_points - (grad_points . points) * points
    grad_tangent = grad_points - np.sum(grad_points * points, axis=1, keepdims=True) * points
    
    # Unflatten
    grad = grad_tangent.flatten()
    
    return smooth_val, grad

def _fibonacci_sphere(n):
    """Generate initial points on Fibonacci sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with different initializations
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Different initializations
        if restart == 0:
            points = _fibonacci_sphere(N)
        else:
            np.random.seed(restart * 12345)
            points = _fibonacci_sphere(N)
            points = points + np.random.randn(N, 3) * 0.1
            points = _normalize(points)
        
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # L-BFGS-B optimization with smooth objective
        params = points.flatten()
        
        # Optimize with increasing beta for sharper approximation
        for beta in [10.0, 30.0, 100.0, 300.0]:
            if time.time() - start_time > time_limit:
                break
            
            def obj_and_grad(p):
                val, grad = _smooth_objective(p, beta)
                return val, grad
            
            try:
                result = minimize(
                    obj_and_grad,
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                params = result.x
            except Exception:
                pass
            
            points = _normalize(params.reshape(N, 3))
            current_radius = _covering_radius(points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Final polish: direct optimization of exact covering radius
        if time.time() - start_time > time_limit:
            break
        
        # Use the smooth objective with very high beta as a proxy for exact
        params = points.flatten()
        
        def obj_and_grad_final(p):
            val, grad = _smooth_objective(p, beta=1000.0)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad_final,
                params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            params = result.x
        except Exception:
            pass
        
        points = _normalize(params.reshape(N, 3))
        current_radius = _covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    return _normalize(best_points)
