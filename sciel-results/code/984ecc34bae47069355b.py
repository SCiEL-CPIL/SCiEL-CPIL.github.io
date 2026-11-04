import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_facet_data(points):
    """Compute facet data: radii, simplices, circumcenters, and vertex-facet incidence."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return None, None, None, None
    
    if len(simplices) == 0:
        return None, None, None, None
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, simplices, u, None

def _soft_max_objective_and_grad(radii, temperature=0.02):
    """
    Compute soft-max surrogate objective and its gradient w.r.t. radii.
    f = temperature * log(sum(exp(radii / temperature)))
    df/dradii = exp(radii / temperature) / sum(exp(radii / temperature))
    """
    shifted = radii - np.max(radii)
    exp_vals = np.exp(shifted / temperature)
    log_sum = np.log(np.sum(exp_vals))
    f = np.max(radii) + temperature * log_sum
    weights = exp_vals / np.sum(exp_vals)
    return f, weights

def _gradient_of_objective_wrt_points(points, radii, simplices, centers, weights):
    """
    Compute gradient of the soft-max objective w.r.t. points on the sphere.
    For each facet m, the gradient contribution is:
      weight_m * d(radii_m)/d(points)
    
    For a facet with vertices i,j,k and circumcenter u_m, circumradius r_m:
      r_m = arccos(u_m · p_i)
      d(r_m)/d(p_i) = -u_m / sin(r_m)
    
    So the total gradient at point i is:
      sum over facets m containing i of: weight_m * (-u_m / sin(r_m))
    
    Then project to tangent space: grad_tangent = grad - (grad · p) * p
    """
    grad = np.zeros_like(points)
    sin_radii = np.sin(radii)
    sin_radii = np.maximum(sin_radii, 1e-10)
    
    for m in range(len(simplices)):
        w_m = weights[m]
        if w_m < 1e-10:
            continue
        u_m = centers[m]
        sin_r_m = sin_radii[m]
        coeff = -w_m / sin_r_m
        
        i_idx, j_idx, k_idx = simplices[m]
        grad[i_idx] += coeff * u_m
        grad[j_idx] += coeff * u_m
        grad[k_idx] += coeff * u_m
    
    # Project to tangent space
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot * points
    
    return grad_tangent

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _polish_worst_facets(points, max_iter=200, tol=1e-12):
    """
    Local polishing: focus on the worst facets and optimize their vertices.
    Uses a dedicated gradient descent on the worst facet's circumradius.
    """
    best_points = points.copy()
    best_radius = np.pi
    
    for it in range(max_iter):
        radii, simplices, centers, _ = _compute_facet_data(points)
        if radii is None:
            break
        
        max_radius = np.max(radii)
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
        
        if max_radius < 0.1974:  # ~11.31 degrees in radians
            break
        
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_r = radii[worst_idx]
        
        if worst_r < 0.1974:
            break
        
        # Get the worst facet's vertices and center
        i_idx, j_idx, k_idx = simplices[worst_idx]
        u_worst = centers[worst_idx]
        
        # Compute gradient for just this facet
        sin_r = np.sin(worst_r)
        if sin_r < 1e-10:
            break
        coeff = -1.0 / sin_r
        
        grad = np.zeros_like(points)
        grad[i_idx] += coeff * u_worst
        grad[j_idx] += coeff * u_worst
        grad[k_idx] += coeff * u_worst
        
        # Project to tangent space
        dot = np.sum(grad * points, axis=1, keepdims=True)
        grad_tangent = grad - dot * points
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-15:
            break
        
        # Try a few step sizes
        best_step = None
        best_new_radius = worst_r
        for alpha in [0.01, 0.005, 0.002, 0.001, 0.0005]:
            new_points = points + alpha * grad_tangent
            new_points = _normalize(new_points)
            
            new_radii, new_simplices, new_centers, _ = _compute_facet_data(new_points)
            if new_radii is None:
                continue
            new_max = np.max(new_radii)
            if new_max < best_new_radius:
                best_new_radius = new_max
                best_step = alpha
        
        if best_step is None:
            break
        
        points = points + best_step * grad_tangent
        points = _normalize(points)
        
        if abs(worst_r - best_new_radius) < tol:
            break
    
    return best_points

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_points = None
    best_radius = np.pi
    
    # Multiple initializations
    inits = []
    
    # Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Random initialization
    rng = np.random.default_rng(42)
    rand_pts = rng.standard_normal((N, 3))
    rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
    inits.append(rand_pts)
    
    # Another structured init: icosahedron-based
    # Start with 12 icosahedron vertices, then add more
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 118 more points via Fibonacci on remaining
    remaining = N - 12
    i_rem = np.arange(remaining) + 0.5
    z_rem = 1.0 - 2.0 * i_rem / remaining
    phi_rem = np.pi * (1.0 + np.sqrt(5.0)) * i_rem
    r_rem = np.sqrt(1.0 - z_rem * z_rem)
    extra = np.column_stack([r_rem * np.cos(phi_rem), r_rem * np.sin(phi_rem), z_rem])
    ico_full = np.vstack([ico, extra])
    inits.append(ico_full)
    
    for init in inits:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        points = _normalize(init.copy())
        
        # Phase 1: Soft-max gradient descent with trust region
        max_iter = 500
        temperature = 0.02
        alpha = 0.05
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            radii, simplices, centers, _ = _compute_facet_data(points)
            if radii is None:
                break
            
            max_radius = np.max(radii)
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
            
            # Compute soft-max objective and gradient
            f, weights = _soft_max_objective_and_grad(radii, temperature)
            grad_tangent = _gradient_of_objective_wrt_points(points, radii, simplices, centers, weights)
            
            # Trust region: limit step size based on gradient norm
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-15:
                break
            
            # Adaptive step: use a fraction of the step that would move to the minimum
            # of the quadratic model
            step = alpha * grad_tangent / grad_norm
            
            # Try the step
            new_points = points + step
            new_points = _normalize(new_points)
            
            new_radii, new_simplices, new_centers, _ = _compute_facet_data(new_points)
            if new_radii is None:
                alpha *= 0.5
                continue
            
            new_max = np.max(new_radii)
            
            # Accept if improved, else reduce step
            if new_max < max_radius:
                points = new_points
                alpha = min(alpha * 1.1, 0.1)
            else:
                alpha *= 0.5
                if alpha < 1e-6:
                    break
        
        # Phase 2: Polish worst facets
        if time.time() - start_time < time_limit - 5.0:
            points = _polish_worst_facets(points, max_iter=100)
            radii, _, _, _ = _compute_facet_data(points)
            if radii is not None:
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
    
    if best_points is None:
        best_points = _normalize(inits[0])
    
    return best_points
