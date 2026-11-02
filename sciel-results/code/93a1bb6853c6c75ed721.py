import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 130
DEADLINE_MARGIN = 15.0

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _covering_radius_hull(points):
    """
    Compute the covering radius in radians using the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        facet_radius = max(d1, d2, d3)
        
        # Check if empty cap
        dots = points @ normal
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        max_dot_other = np.max(dots[mask])
        
        if max_dot_other > np.cos(facet_radius) + 1e-12:
            continue
            
        max_angle = max(max_angle, facet_radius)
        
    return max_angle

def _lloyd_iteration(points, n_iters=50, deadline=None):
    """
    Lloyd's algorithm on the sphere:
    1. Compute Voronoi cells (via Delaunay / Convex Hull)
    2. Move each point to the centroid (spherical mean) of its cell
    """
    points = points.copy().astype(float)
    for _ in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        circumcenters = []
        facet_vertices = []
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            circumcenters.append(normal)
            facet_vertices.append(simplex)
        
        if len(circumcenters) == 0:
            break
            
        circumcenters = np.array(circumcenters)
        
        new_points = np.zeros_like(points)
        
        for i in range(N):
            mask = [idx for idx, simplex in enumerate(facet_vertices) if i in simplex]
            if len(mask) == 0:
                new_points[i] = points[i]
                continue
            
            cc = circumcenters[mask]
            
            mean = np.sum(cc, axis=0)
            norm = np.linalg.norm(mean)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = mean / norm
        
        points = new_points
        
    return points

def _smooth_covering_radius(points, grid_points, beta=30.0):
    """
    Smooth approximation of covering radius using log-sum-exp.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each grid point, find max dot product (min angular distance)
    max_dots = dots.max(axis=1)
    max_dots = np.clip(max_dots, -0.999999, 0.999999)
    distances = np.arccos(max_dots)
    
    # Log-sum-exp for smooth max
    scaled = beta * distances
    max_scaled = scaled.max()
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    return log_sum_exp / beta

def _gradient_optimize(points, deadline=None):
    """
    Optimize using gradient descent on a smooth approximation of the covering radius.
    """
    n_theta = 120
    n_phi = 240
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    beta = 30.0
    
    def objective(flat_params):
        p = flat_params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        return _smooth_covering_radius(p, grid_points, beta)
    
    def gradient(flat_params):
        p = flat_params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        
        dots = grid_points @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        
        max_dots = dots.max(axis=1)
        max_dots = np.clip(max_dots, -0.999999, 0.999999)
        distances = np.arccos(max_dots)
        
        # Soft weights for which point is nearest
        gamma = 50.0
        exp_dots = np.exp(gamma * dots)
        sum_exp_dots = exp_dots.sum(axis=1, keepdims=True)
        weights = exp_dots / sum_exp_dots
        
        # Gradient of arccos(dot) w.r.t. point position
        # d(arccos(d))/d(d) = -1/sqrt(1-d^2)
        # d(dot)/d(p_i) = grid_point
        inv_sqrt = 1.0 / np.sqrt(np.clip(1.0 - max_dots**2, 1e-10, 1.0))
        
        # For each point i, gradient is sum over grid points of weight * (-inv_sqrt) * grid_point
        # Then project to tangent plane
        grad_p = np.zeros((N, 3))
        for i in range(N):
            w_i = weights[:, i] * inv_sqrt
            grad_p[i] = -(grid_points.T @ w_i)
        
        # Project to tangent plane
        for i in range(N):
            radial = np.dot(grad_p[i], p[i]) * p[i]
            grad_p[i] = grad_p[i] - radial
        
        return grad_p.ravel()
    
    x0 = points.ravel()
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 30, 'gtol': 1e-4, 'maxfun': 100}
        )
        if res.success or res.fun < objective(x0):
            new_points = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def _nelder_mead_polish(points, deadline=None):
    """
    Nelder-Mead local optimization to polish the solution.
    """
    def objective(flat_params):
        p = flat_params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        return _covering_radius_hull(p)
    
    x0 = points.ravel()
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 200}
        )
        if res.fun < objective(x0):
            new_points = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    inits = []
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configurations
    for pts in inits:
        r = _covering_radius_hull(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 20:
            break
            
    # Optimize using Lloyd's algorithm
    if best_points is not None and time.time() < deadline - 10:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=80, deadline=deadline - 10)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
    # Try gradient optimization
    if best_points is not None and time.time() < deadline - 15:
        grad_pts = _gradient_optimize(best_points, deadline=deadline - 10)
        r = _covering_radius_hull(grad_pts)
        if r < best_radius:
            best_radius = r
            best_points = grad_pts.copy()
            
    # Try optimizing from other initial configs
    for pts in inits:
        if time.time() > deadline - 15:
            break
        if np.allclose(pts, best_points, atol=1e-6):
            continue
            
        lloyd_pts = _lloyd_iteration(pts, n_iters=40, deadline=deadline - 10)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
            if time.time() < deadline - 15:
                grad_pts = _gradient_optimize(best_points, deadline=deadline - 10)
                r = _covering_radius_hull(grad_pts)
                if r < best_radius:
                    best_radius = r
                    best_points = grad_pts.copy()
                    
    # Final refinement with Nelder-Mead
    if best_points is not None and time.time() < deadline - 5:
        nm_pts = _nelder_mead_polish(best_points, deadline=deadline - 2)
        r = _covering_radius_hull(nm_pts)
        if r < best_radius:
            best_points = nm_pts.copy()
            
    # Final Lloyd's pass
    if best_points is not None and time.time() < deadline - 2:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=15, deadline=deadline - 1)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_points = lloyd_pts.copy()

    if best_points is None:
        best_points = inits[0]
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius_hull(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
