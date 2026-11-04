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

def _repulsion_energy(points, k=10.0):
    """
    Compute repulsion energy to prevent point clustering.
    Uses inverse distance potential.
    """
    n = len(points)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d > 1e-10:
                energy += k / (d ** 2)
    return energy

def _smooth_facet_surrogate(params, beta=50.0, repulsion_weight=0.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet circumradii.
    Optionally add repulsion energy to prevent clustering.
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
    result = logsumexp / beta
    
    # Add repulsion energy if requested
    if repulsion_weight > 0:
        result += repulsion_weight * _repulsion_energy(pts)
    
    return result

def _gradient_descent_phase(points, time_limit=20.0, beta=50.0, repulsion_weight=0.0):
    """
    Perform gradient descent using L-BFGS-B on the smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_facet_surrogate(params, beta=beta, repulsion_weight=repulsion_weight)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points

def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((N, 3))
    return _normalize_points(points)

def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with jitter."""
    t = (1 + np.sqrt(5)) / 2
    base = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    base = _normalize_points(base)
    
    # Add jittered copies to reach N points
    rng = np.random.default_rng(123)
    extra = rng.standard_normal((N - 12, 3))
    extra = _normalize_points(extra)
    # Add small perturbation to base points
    base_jittered = base + rng.standard_normal((12, 3)) * 0.1
    base_jittered = _normalize_points(base_jittered)
    
    return np.vstack([base_jittered, extra])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Try multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_points()),
        ("random1", _random_points(42)),
        ("random2", _random_points(123)),
        ("icosahedron", _icosahedron_based_points()),
    ]
    
    best_points = None
    best_radius = np.pi
    
    for name, init_pts in initializations:
        if time.time() > deadline - 40.0:
            break
            
        current_radius, _, _ = _compute_covering_radius(init_pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = init_pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_radius, _, _ = _compute_covering_radius(best_points)
    
    # Phase 1: Multi-stage gradient descent with increasing beta
    # Start with low beta for smooth landscape, then increase
    beta_schedule = [10, 30, 100, 300, 1000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
            
        time_limit = 15.0 if beta <= 100 else 10.0
        # Add small repulsion weight for early stages to prevent clustering
        rep_w = 0.01 if beta <= 30 else 0.0
        
        points = _gradient_descent_phase(best_points, time_limit=time_limit, beta=beta, repulsion_weight=rep_w)
        current_radius, _, _ = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Geometric Relaxation (Local Polish)
    # This is the "facet shrinking" method.
    # Use adaptive step size with more iterations
    
    step = 0.03
    prev_radius = best_radius
    
    for it in range(5000):
        if time.time() > deadline - 3.0:
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_radius(best_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
        
        if worst_facet is None:
            break
        
        # Adaptive step: reduce if radius increased, increase slightly if decreased
        if current_radius > prev_radius + 1e-10:
            step *= 0.8
        elif current_radius < prev_radius - 1e-10:
            step = min(step * 1.05, 0.05)
        prev_radius = current_radius
        
        # Move the vertices of the worst facet towards the circumcenter
        v0 = best_points[worst_facet[0]]
        v1 = best_points[worst_facet[1]]
        v2 = best_points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            # Projection of c onto tangent space at v
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return _normalize(v_new)
        
        # Update vertices
        best_points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        best_points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        best_points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        # Very slow decay to allow fine-tuning
        if it % 500 == 0:
            step *= 0.98
            if step < 0.0005:
                step = 0.0005
    
    # Phase 3: Final high-precision polish with very high beta
    if time.time() < deadline - 5.0:
        points = _gradient_descent_phase(best_points, time_limit=8.0, beta=5000, repulsion_weight=0.0)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Final check
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return _fibonacci_points()
# EVOLVE-BLOCK-END
