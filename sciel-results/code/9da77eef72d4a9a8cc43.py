"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _get_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    """
    n = np.cross(p1 - p2, p2 - p3)
    c = _normalize(n)
    # Ensure c is on the same side as the vertices (dot product positive)
    if np.dot(c, p1) < 0:
        c = -c
    cos_r = np.dot(c, p1)
    # Clamp for numerical stability
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c

def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration in radians.
    Returns max_r, worst_facet_indices, worst_center.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, c = _get_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = c
        return max_r, worst_facet, worst_center
    except Exception:
        # If convex hull fails, return a large value
        return np.pi, None, None

def _smooth_facet_surrogate(params, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet circumradii.
    This provides a gradient for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi

    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Normal to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Handle degenerate triangles
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n / norm_n
    
    # Ensure normals point towards the interior of the sphere relative to the triangle?
    # The circumcenter c is parallel to n.
    # We need c.p1 > 0.
    dots = np.sum(n * p1, axis=1)
    n[dots < 0] *= -1
    
    # Calculate circumradius
    # cos(R) = c.p1
    cos_r = np.sum(n * p1, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # Soft-max (log-sum-exp)
    # We want to minimize the max radius, so we approximate max(radii) with logsumexp(beta * radii) / beta
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta

def _gradient_descent_phase(points, time_limit=20.0, beta=50.0):
    """
    Perform gradient descent using L-BFGS-B on the smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_facet_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 200})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points

def _repulsion_energy(points, exponent=6.0):
    """Compute repulsion energy to encourage spread-out points."""
    n = len(points)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                energy += 1e10
            else:
                energy += 1.0 / (d ** exponent)
    return energy

def _repulsion_gradient(points, exponent=6.0):
    """Compute gradient of repulsion energy."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            # d/dx_i (1/|x_i - x_j|^exponent) = -exponent * diff / |x_i - x_j|^(exponent+2)
            coeff = -exponent / (d ** (exponent + 2))
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad

def _repulsion_refine(points, time_limit=15.0, exponent=6.0):
    """Refine points using repulsion energy to get better initial spread."""
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = _normalize_points(params.reshape(N, 3))
        return _repulsion_energy(pts, exponent)
    
    def gradient(params):
        pts = _normalize_points(params.reshape(N, 3))
        g = _repulsion_gradient(pts, exponent)
        # Project gradient to tangent space
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 100})
        return _normalize_points(result.x.reshape(N, 3))
    except Exception:
        return points

def _multi_beta_lse_refine(points, time_budget=60.0, betas=[10, 50, 200, 500]):
    """Multi-stage beta cascade for LSE refinement."""
    start_time = time.time()
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
        points = _gradient_descent_phase(points, time_limit=remaining * 0.8, beta=beta)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Generate multiple initial configurations
    best_points = None
    best_radius = np.pi
    
    # Configuration 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Configuration 2: Jittered Fibonacci
    rng = np.random.RandomState(42)
    jitter = 0.1 * rng.randn(N, 3)
    jittered_fib = _normalize_points(fib_points + jitter)
    
    # Configuration 3: Random points
    random_pts = _normalize_points(rng.randn(N, 3))
    
    # Configuration 4: Another Fibonacci variant
    i2 = np.arange(N)
    z2 = 1.0 - 2.0 * (i2 + 0.3) / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 * 1.1
    r2 = np.sqrt(1.0 - z2 * z2)
    fib2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    
    initial_configs = [fib_points, jittered_fib, random_pts, fib2]
    
    # Phase 0: Repulsion refinement on each initial config
    for config in initial_configs:
        if time.time() > deadline - 80.0:
            break
        refined = _repulsion_refine(config, time_limit=10.0)
        radius, _, _ = _compute_covering_radius(refined)
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    if best_points is None:
        best_points = fib_points
        best_radius, _, _ = _compute_covering_radius(best_points)
    
    # Phase 1: Multi-beta LSE refinement
    if time.time() < deadline - 40.0:
        refined = _multi_beta_lse_refine(best_points, time_budget=35.0, betas=[10, 50, 200, 500])
        current_radius, _, _ = _compute_covering_radius(refined)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = refined.copy()
    
    # Phase 2: Geometric Relaxation (Local Polish)
    points = best_points.copy()
    step = 0.03
    
    for it in range(3000):
        if time.time() > deadline - 5.0:
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if worst_facet is None:
            break
            
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return _normalize(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        if it % 100 == 0:
            step *= 0.95
            if step < 0.001:
                step = 0.001
    
    # Final high-precision polish with very small steps
    if time.time() < deadline - 3.0:
        step = 0.005
        for it in range(500):
            if time.time() > deadline - 3.0:
                break
            current_radius, worst_facet, worst_center = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            if worst_facet is None:
                break
            v0 = points[worst_facet[0]]
            v1 = points[worst_facet[1]]
            v2 = points[worst_facet[2]]
            
            def move_vertex(v, c, step):
                grad = c - np.dot(c, v) * v
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    return v
                grad = grad / grad_norm
                v_new = v + step * grad
                return _normalize(v_new)
            
            points[worst_facet[0]] = move_vertex(v0, worst_center, step)
            points[worst_facet[1]] = move_vertex(v1, worst_center, step)
            points[worst_facet[2]] = move_vertex(v2, worst_center, step)
    
    # Final check
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return _normalize_points(fib_points)
# EVOLVE-BLOCK-END
