import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _calc_covering_radius(pts):
    """Calculate the covering radius from the convex hull facets."""
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        max_dist = 0.0
        for s in simplices:
            v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if np.dot(n, v1) < 0:
                n = -n
            dot_val = np.dot(n, v1)
            if abs(dot_val) < 1e-10:
                continue
            k = 1.0 / (2.0 * dot_val)
            c = k * n
            c = c / np.linalg.norm(c)
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if dist > max_dist:
                max_dist = dist
        return max_dist
    except Exception:
        return np.inf

def _compute_facet_data_vec(points):
    """Vectorized computation of circumcenters and distances for all facets."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        v1 = points[simplices[:, 0]]
        v2 = points[simplices[:, 1]]
        v3 = points[simplices[:, 2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        n_norms = np.linalg.norm(n, axis=1)
        
        # Mask out degenerate facets
        valid = n_norms > 1e-10
        if not np.any(valid):
            return np.array([]), np.array([]), []
        
        n = n[valid] / n_norms[valid, np.newaxis]
        v1v = v1[valid]
        
        # Ensure normals point outward
        dots = np.sum(n * v1v, axis=1)
        n[dots < 0] *= -1
        dots = np.sum(n * v1v, axis=1)
        
        valid2 = np.abs(dots) > 1e-10
        if not np.any(valid2):
            return np.array([]), np.array([]), []
        
        n = n[valid2]
        v1v = v1v[valid2]
        dots = dots[valid2]
        
        k = 1.0 / (2.0 * dots)
        c = k[:, np.newaxis] * n
        c_norms = np.linalg.norm(c, axis=1)
        valid3 = c_norms > 1e-10
        if not np.any(valid3):
            return np.array([]), np.array([]), []
        
        c = c[valid3] / c_norms[valid3, np.newaxis]
        v1v = v1v[valid3]
        
        dists = np.arccos(np.clip(np.sum(c * v1v, axis=1), -1.0, 1.0))
        
        # Get corresponding facet indices
        valid_mask = np.zeros(F, dtype=bool)
        valid_mask[np.where(valid)[0][valid2][valid3]] = True
        facet_indices = simplices[valid_mask].tolist()
        
        return c, dists, facet_indices
    except Exception:
        return np.array([]), np.array([]), []

def _smooth_covering_radius(points, beta=20.0):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    Returns (smooth_value, gradient) where gradient is w.r.t. point positions.
    """
    c, dists, facet_indices = _compute_facet_data_vec(points)
    if len(c) == 0:
        return np.inf, np.zeros_like(points)
    
    max_d = np.max(dists)
    # Log-sum-exp for numerical stability
    shifted = beta * (dists - max_d)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_d * beta
    smooth_val = log_sum / beta
    
    # Weights for each facet
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Compute gradient: d/d(point_j) of sum_i w_i * dist_i
    # dist_i = arccos(dot(c_i, v_i)) where c_i is circumcenter of facet i
    # d(dist_i)/d(point_j) involves the derivative of the circumcenter w.r.t. points
    # For now, use the simpler target-averaging approach as gradient proxy
    
    grad = np.zeros_like(points)
    for i, s in enumerate(facet_indices):
        w = weights[i]
        c_i = c[i]
        # Gradient contribution: move points toward circumcenter weighted by w
        grad[s[0]] += w * (c_i - points[s[0]])
        grad[s[1]] += w * (c_i - points[s[1]])
        grad[s[2]] += w * (c_i - points[s[2]])
    
    return smooth_val, grad

def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize(points, start_time, time_limit, rng):
    """Run local optimization using smooth surrogate with adaptive beta."""
    points = points.copy()
    best_points = points.copy()
    best_radius = _calc_covering_radius(points)
    
    momentum = np.zeros_like(points)
    lr = 0.05
    momentum_factor = 0.7
    iterations = 0
    no_improve_count = 0
    patience = 200
    beta = 5.0
    
    while time.time() - start_time < time_limit:
        iterations += 1
        
        try:
            smooth_val, grad = _smooth_covering_radius(points, beta)
            if np.isinf(smooth_val):
                break
            
            # Project gradient onto tangent space of sphere
            for j in range(N):
                grad[j] -= np.dot(grad[j], points[j]) * points[j]
            
            momentum = momentum_factor * momentum + lr * grad
            points += momentum
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        except Exception:
            break

        current_radius = _calc_covering_radius(points)
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = points.copy()
            no_improve_count = 0
            # Increase beta as we converge to focus on worst facets
            beta = min(50.0, beta * 1.02)
        else:
            no_improve_count += 1
        
        # Adaptive learning rate
        lr *= 0.9995
        
        # Periodic noise to escape local minima
        if iterations % 100 == 0:
            noise_scale = 0.02
            noise = rng.standard_normal(points.shape) * noise_scale
            for j in range(N):
                noise[j] -= np.dot(noise[j], points[j]) * points[j]
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0
        
        # Aggressive shake when stuck
        if no_improve_count > patience:
            noise_scale = 0.05
            noise = rng.standard_normal(points.shape) * noise_scale
            for j in range(N):
                noise[j] -= np.dot(noise[j], points[j]) * points[j]
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0
            lr = 0.03
            beta = max(5.0, beta * 0.5)  # Reduce beta to explore
        
        if no_improve_count > patience * 5:
            break
    
    return best_points, best_radius

def _polish(points, start_time, time_limit, rng):
    """High-precision polish pass with very small steps and high beta."""
    points = points.copy()
    best_points = points.copy()
    best_radius = _calc_covering_radius(points)
    
    momentum = np.zeros_like(points)
    lr = 0.002
    beta = 30.0
    polish_start = time.time()
    
    for _ in range(10000):
        if time.time() - start_time >= time_limit - 2:
            break
        if time.time() - polish_start > 15:
            break
        
        try:
            smooth_val, grad = _smooth_covering_radius(points, beta)
            if np.isinf(smooth_val):
                break
            
            for j in range(N):
                grad[j] -= np.dot(grad[j], points[j]) * points[j]
            
            momentum = 0.8 * momentum + lr * grad
            points += momentum
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        except Exception:
            break

        current_radius = _calc_covering_radius(points)
        if current_radius < best_radius - 1e-12:
            best_radius = current_radius
            best_points = points.copy()
            beta = min(100.0, beta * 1.05)
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s safety margin for 120s limit
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time >= time_limit - 10:
            break
        
        if restart == 0:
            points = _fibonacci_sphere(N)
        elif restart == 1:
            points = _fibonacci_sphere(N)
            noise = rng.standard_normal(points.shape) * 0.08
            points += noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = _fibonacci_sphere(N)
            noise = rng.standard_normal(points.shape) * 0.15
            points += noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            points = _random_sphere(N, rng)
        else:
            w = 0.3 + 0.4 * rng.random()
            points = _random_sphere(N, rng)
            fib = _fibonacci_sphere(N)
            points = w * points + (1 - w) * fib
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        opt_points, opt_radius = _optimize(points, start_time, time_limit, rng)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
        
        if best_radius < np.radians(18.35):
            break
    
    # Final polish
    if best_points is not None and time.time() - start_time < time_limit - 3:
        polished, pol_radius = _polish(best_points, start_time, time_limit, rng)
        if pol_radius < best_radius:
            best_points = polished
            best_radius = pol_radius
    
    return best_points
