import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    This is used for the final score and as a fallback.
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return np.pi
        
    simplices = hull.simplices
    max_radius = 0.0
    
    for tri in simplices:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-10:
            center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-10:
                radius = np.pi
            else:
                center = center / norm_c
                cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
                radius = np.arccos(cos_r)
        else:
            center = cross / norm_cross
            centroid = (a + b + c) / 3.0
            if np.dot(center, centroid) < 0:
                center = -center
            
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radius = np.arccos(cos_r)
            
        if radius > max_radius:
            max_radius = radius
            
    return max_radius

def _find_worst_point(pts, sigma, grid_size=5000):
    """
    Find the point y on the sphere that minimizes the potential sum.
    Uses a coarse grid followed by local refinement.
    """
    # Generate a Fibonacci sphere grid for sampling
    i = np.arange(grid_size) + 0.5
    z = 1.0 - 2.0 * i / grid_size
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute potential at grid points
    # Vectorized dot products
    # dists_sq = 1 - cos(theta) = 1 - grid @ pts.T
    cos_theta = np.clip(grid @ pts.T, -1.0, 1.0)
    # Potential: exp((cos_theta - 1) / sigma^2)
    # To avoid underflow/overflow issues with large sigma or small sigma, 
    # we can work with log-potential or just standard exp if sigma is reasonable.
    # sigma ~ 0.2 rad -> sigma^2 ~ 0.04. cos-1 is in [-2, 0].
    # Exponent is in [-50, 0]. exp(-50) is ~ 1e-22. Safe.
    pot = np.exp((cos_theta - 1.0) / (sigma * sigma))
    total_pot = np.sum(pot, axis=1)
    
    min_idx = np.argmin(total_pot)
    y_min = grid[min_idx]
    
    # Local refinement around y_min
    # We want to maximize -total_pot(y)
    def neg_pot(y):
        y = y / np.linalg.norm(y)
        cos_y = np.clip(y @ pts.T, -1.0, 1.0)
        p = np.exp((cos_y - 1.0) / (sigma * sigma))
        return -np.sum(p)
        
    res = minimize(neg_pot, y_min, method='BFGS', options={'gtol': 1e-8, 'maxiter': 100})
    if res.success:
        y_min = res.x / np.linalg.norm(res.x)
        
    return y_min

def _potential_gradient_step(pts, sigma, step_size):
    """
    Perform a gradient ascent step on the points to maximize the minimum potential.
    """
    y_min = _find_worst_point(pts, sigma)
    
    # Gradient of potential at y_min w.r.t each p_i
    # U(y) = sum exp((cos(y, p_i) - 1) / sigma^2)
    # d/dp_i exp(...) = exp(...) * (1/sigma^2) * d/dp_i (cos(y, p_i) - 1)
    # d/dp_i cos(y, p_i) = - (y - (y.p_i)p_i) / |p_i|^2 ? No.
    # cos(y, p) = y . p.
    # grad_p (y . p) = y.
    # But we are on the sphere. The gradient on the manifold is the projection of the ambient gradient.
    # Ambient grad w.r.t p_i:
    # term_i = exp((cos(y, p_i) - 1) / sigma^2)
    # grad_ambient_i = term_i * (1/sigma^2) * y
    
    # Wait, cos(y, p) = y . p.
    # d/dp (y . p) = y.
    # So grad_ambient = sum_i term_i * (1/sigma^2) * y ?
    # No, the gradient of the sum is the sum of the gradients.
    # The gradient w.r.t p_i is just the term for that p_i.
    
    cos_y_p = np.clip(y_min @ pts.T, -1.0, 1.0)
    terms = np.exp((cos_y_p - 1.0) / (sigma * sigma))
    
    # Ambient gradient for each point i:
    # g_i = terms[i] * (1/sigma^2) * y_min
    # This is the same direction for all points?
    # No, the potential is sum of kernels centered at p_i.
    # The contribution of p_i to the gradient at y_min is proportional to the kernel value at y_min.
    # So points closer to y_min (higher kernel value) should move more?
    # Actually, moving p_i towards y_min increases the kernel value at y_min.
    # The direction to move p_i is towards y_min.
    # The magnitude is weighted by the kernel value.
    
    # Riemannian gradient: Project y_min onto tangent space of p_i.
    # t_i = y_min - (y_min . p_i) p_i
    # t_i = y_min - cos(y_min, p_i) p_i
    
    grad = np.zeros_like(pts)
    for i in range(N):
        t_i = y_min - cos_y_p[i] * pts[i]
        # Normalize t_i? No, the gradient magnitude is part of the step.
        # But we want to move towards y_min.
        # The "pull" is stronger if the kernel is larger (point is closer to y_min).
        # So grad_i = terms[i] * t_i
        
        # Wait, if p_i is far from y_min, terms[i] is small, so it moves little.
        # If p_i is close to y_min, terms[i] is large, so it moves a lot towards y_min?
        # That would collapse them onto y_min.
        
        # Let's re-evaluate.
        # We want to increase U(y_min).
        # U(y_min) = sum K(y_min, p_i).
        # d U / d p_i = grad_{p_i} K(y_min, p_i).
        # K(y, p) = exp((y.p - 1)/sigma^2).
        # grad_p K = K * (1/sigma^2) * y.
        # So the ambient gradient is K(y_min, p_i) * (1/sigma^2) * y_min.
        # This vector points towards y_min.
        # Projecting to tangent space:
        # g_i = K(y_min, p_i) * (1/sigma^2) * (y_min - (y_min.p_i) p_i).
        
        # This means points close to y_min have large K, so they have large gradient.
        # They move towards y_min.
        # Points far away have small K, so they move little.
        # This seems to pull the cluster towards y_min, which is good for increasing the minimum at y_min.
        
        grad[i] = terms[i] * t_i
        
    # Normalize the total gradient? Or just use step size?
    # The magnitudes vary. Let's normalize each gradient vector to unit length?
    # No, that ignores the weight.
    # Let's scale the step by the average term magnitude or similar.
    # Or just use a small step size.
    
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            # Normalize direction
            g_dir = grad[i] / g_norm
            # Step size adjustment?
            # If we just use step_size, the effective step is step_size * (terms[i] * |t_i|).
            # terms[i] can be very small.
            # Maybe normalize the gradient vector to have unit norm, but keep the direction?
            # That would treat all points equally in terms of step size, but direction depends on weight?
            # No, direction is always towards y_min (projected).
            # So all points move towards y_min.
            # That is correct.
            
            # However, if we normalize g_dir, we lose the "pull" strength information.
            # But for a "worst point" update, we just want to move towards it.
            # Let's use the normalized direction.
            
            new_pts[i] = pts[i] + step_size * g_dir
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine_potential(pts, deadline, max_iters=500, initial_step=0.05, sigma=0.3):
    """
    Local refinement using the smooth potential method.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _potential_gradient_step(pts, sigma, step_size)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            step_size *= 0.99
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6 or no_improve_count > 50:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Try different sigmas
    sigmas = [0.2, 0.3, 0.5, 0.8]
    
    for sigma in sigmas:
        if time.time() > deadline - 20:
            break
            
        refined_pts, refined_radius = _local_refine_potential(points.copy(), deadline - 20, max_iters=200, initial_step=0.05, sigma=sigma)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Final polish with the best sigma (or just use the last one if it improved)
    # Let's do a few restarts with the potential method
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        pts = best_points + np.random.randn(N, 3) * 0.05
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        refined_pts, refined_radius = _local_refine_potential(pts, deadline - 5, max_iters=100, initial_step=0.02, sigma=0.3)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
