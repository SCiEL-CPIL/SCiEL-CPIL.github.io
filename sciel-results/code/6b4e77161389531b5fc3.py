import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _compute_facet_data(points):
    """
    Compute circumcenters and angular radii for all facets of the convex hull.
    Returns:
        centers: (M, 3) array of circumcenter unit vectors
        radii: (M,) array of angular radii in radians
        simplices: (M, 3) array of facet indices
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    # Vectorized computation
    v0 = points[facets[:, 0]]
    v1 = points[facets[:, 1]]
    v2 = points[facets[:, 2]]
    
    # Normal to each facet plane
    e1 = v1 - v0
    e2 = v2 - v1
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Avoid division by zero
    small = norms < 1e-10
    norms[small] = 1.0
    centers = normals / norms
    
    # Determine correct orientation using hull equations
    # hull.equations[i] gives [a,b,c,d] where (a,b,c) is outward normal
    hull_normals = hull.equations[:, :3]
    
    # Check dot product; if negative, flip
    dots = np.sum(centers * hull_normals, axis=1)
    flip = dots < 0
    centers[flip] *= -1
    
    # Angular radius: arccos(dot(center, v0))
    dot_vals = np.sum(centers * v0, axis=1)
    dot_vals = np.clip(dot_vals, -1.0, 1.0)
    radii = np.arccos(dot_vals)
    
    # Handle degenerate facets
    radii[small[:, 0]] = 0.0
    centers[small[:, 0]] = v0[small[:, 0]]
    
    return centers, radii, facets

def _covering_radius(points):
    """Compute the covering radius in radians."""
    centers, radii, _ = _compute_facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _soft_max_radius_and_grad(points, beta=100.0):
    """
    Compute soft-max of facet radii and its gradient.
    Uses log-sum-exp for numerical stability.
    Returns:
        soft_max: scalar
        grad: (N, 3) gradient w.r.t. points
    """
    centers, radii, facets = _compute_facet_data(points)
    if radii is None:
        return np.pi, np.zeros_like(points)
    
    M = len(radii)
    
    # Log-sum-exp for soft max
    max_r = np.max(radii)
    if max_r < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # weights = exp(beta * (r_i - r_max))
    exp_args = beta * (radii - max_r)
    exp_args = np.clip(exp_args, -100, 100)  # avoid underflow issues
    weights = np.exp(exp_args)
    sum_weights = np.sum(weights)
    
    if sum_weights < 1e-30:
        return max_r, np.zeros_like(points)
    
    soft_max = max_r + np.log(sum_weights) / beta
    
    # Gradient computation
    # d(soft_max)/d(points_j) = sum_i (w_i / sum_w) * d(r_i)/d(points_j)
    # r_i = arccos(dot(c_i, v_{f0}))
    # d(r_i)/dv = -c_i / sqrt(1 - dot(c_i, v)^2) for the vertex v on the facet
    # But we need to be careful: the circumcenter c_i depends on the points too.
    # For simplicity, we approximate by treating c_i as fixed (first-order).
    # This is a common approximation in such problems.
    
    grad = np.zeros_like(points)
    
    # For each facet, compute contribution
    # d(r_i)/d(v_k) for k in facet i
    # r_i = arccos(d_i) where d_i = dot(c_i, v_{i0})
    # d(r_i)/d(v_{i0}) = -c_i / sqrt(1 - d_i^2)
    # But actually, since c_i is perpendicular to the facet plane and v_{i0} is on the sphere,
    # the derivative is more complex. Let's use a simpler approach:
    # Move each point in the facet towards the circumcenter to reduce the radius.
    
    for i in range(M):
        w = weights[i] / sum_weights
        if w < 1e-10:
            continue
        
        face = facets[i]
        c = centers[i]
        r = radii[i]
        
        if r < 1e-10:
            continue
            
        for j in face:
            v = points[j]
            # Direction to move: towards circumcenter
            dir_vec = c - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                # The rate of change of arccos(dot(c,v)) w.r.t. moving v towards c
                # is approximately -1/sqrt(1-dot^2) * dot(c, dir_vec/norm_dir)
                dot_cv = np.dot(c, v)
                denom = np.sqrt(max(1e-10, 1.0 - dot_cv * dot_cv))
                factor = 1.0 / denom
                grad[j] -= w * factor * dir_vec / norm_dir
    
    return soft_max, grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return points / norms

def _lbfgs_stage(points, max_iters=200, timeout=30, beta=100.0):
    """
    Run L-BFGS-B optimization on the soft-max objective.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        p = _project_to_sphere(x.reshape(N, 3))
        val, g = _soft_max_radius_and_grad(p, beta=beta)
        return val, g.reshape(-1)
    
    x0 = points.reshape(-1)
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
    )
    
    if time.time() - start_time > timeout:
        return _project_to_sphere(points)
    
    best_x = result.x if result.success or result.fun < np.pi else x0
    return _project_to_sphere(best_x.reshape(N, 3))

def _local_polish(points, max_iters=100, timeout=10):
    """
    Final high-precision polish using gradient descent with decreasing step size.
    """
    start_time = time.time()
    points = points.copy()
    step_size = 0.01
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        val, grad = _soft_max_radius_and_grad(points, beta=200.0)
        
        if np.linalg.norm(grad) < 1e-10:
            break
        
        points += step_size * grad
        points = _project_to_sphere(points)
        
        step_size *= 0.95
        if step_size < 1e-6:
            break
    
    return points

def _generate_seeds(rng, n_seeds=80):
    """Generate diverse initial configurations."""
    seeds = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # Random starts with varying noise
    for s in range(n_seeds - 1):
        p = rng.normal(size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        # Add varying perturbation
        noise_scale = 0.05 * (0.7 ** (s // 10))
        noise = rng.normal(size=(N, 3)) * noise_scale
        p = p + noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        seeds.append(p)
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    best_points = None
    best_radius = np.pi
    
    rng = np.random.default_rng(42)
    seeds = _generate_seeds(rng, n_seeds=80)
    
    for s, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Phase 1: L-BFGS with moderate beta
        p1 = _lbfgs_stage(seed, max_iters=100, timeout=min(remaining * 0.5, 20), beta=50.0)
        
        if time.time() > deadline - 5:
            r = _covering_radius(p1)
            if r < best_radius:
                best_radius = r
                best_points = p1
            continue
        
        # Phase 2: L-BFGS with higher beta
        p2 = _lbfgs_stage(p1, max_iters=150, timeout=min(remaining * 0.3, 15), beta=200.0)
        
        if time.time() > deadline - 3:
            r = _covering_radius(p2)
            if r < best_radius:
                best_radius = r
                best_points = p2
            continue
        
        # Phase 3: Final polish
        p3 = _local_polish(p2, max_iters=50, timeout=min(remaining * 0.2, 5))
        
        r = _covering_radius(p3)
        if r < best_radius:
            best_radius = r
            best_points = p3
    
    if best_points is None:
        # Fallback: return the first seed
        return _project_to_sphere(seeds[0])
    
    return best_points
