import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_circumradius_and_center(a, b, c):
    """
    Compute the circumradius (in radians) and circumcenter of a spherical triangle
    defined by unit vectors a, b, c.
    """
    # Dot products
    ab = np.dot(a, b)
    bc = np.dot(b, c)
    ca = np.dot(c, a)
    
    # Clip for numerical stability
    ab = np.clip(ab, -1.0, 1.0)
    bc = np.clip(bc, -1.0, 1.0)
    ca = np.clip(ca, -1.0, 1.0)
    
    # We want C = x*a + y*b + z*c such that C.a = C.b = C.c = cos(r)
    # This leads to the system:
    # x + y*ab + z*ca = cos(r)
    # x*ab + y + z*bc = cos(r)
    # x*ca + y*bc + z = cos(r)
    
    # We can solve for x, y, z relative to each other.
    # Let's find the direction of C.
    # The circumcenter is proportional to the vector that is equidistant.
    # A known formula for the circumcenter vector direction (before normalization) is:
    # V = a * (1 - bc^2) + b * (1 - ca^2) + c * (1 - ab^2)
    # Let's verify this.
    # If C = V, does C.a = C.b?
    # V.a = (1-bc^2) + ab(1-ca^2) + ca(1-ab^2)
    # V.b = ab(1-bc^2) + (1-ca^2) + bc(1-ab^2)
    # This seems complicated. Let's use the linear system approach which is robust.
    
    # The system M * [x, y, z]^T = cos(r) * [1, 1, 1]^T
    # M = [[1, ab, ca], [ab, 1, bc], [ca, bc, 1]]
    # We can solve for [x, y, z] up to a scale factor if we set cos(r)=1? No, that would be the trivial solution if points are close.
    # Actually, we just need the direction of C.
    # The vector C is in the null space of the matrix formed by differences?
    # Simpler: C is the intersection of the two perpendicular bisectors.
    # Bisector of ab: points p such that p.a = p.b => p.(a-b) = 0.
    # Bisector of bc: points p such that p.b = p.c => p.(b-c) = 0.
    # So C is proportional to (a-b) x (b-c).
    
    v1 = a - b
    v2 = b - c
    cross = np.cross(v1, v2)
    
    # Check if degenerate
    if np.linalg.norm(cross) < 1e-10:
        # Points are collinear or identical. Use centroid as fallback
        center = a + b + c
        norm = np.linalg.norm(center)
        if norm < 1e-10:
            return np.pi, np.array([0,0,0])
        center = center / norm
        cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
        return np.arccos(cos_r), center

    center = cross / np.linalg.norm(cross)
    
    # Determine the sign. The circumcenter should be on the same side of the triangle as the origin?
    # No, for a spherical triangle containing the origin in its interior (on the sphere),
    # the circumcenter is just the normalized cross product.
    # However, if the triangle is large (contains the antipode of the center?),
    # we need to be careful.
    # The covering radius is the distance to the *nearest* point.
    # The circumcenter of a facet of the convex hull is the center of the empty cap.
    # So the circumcenter must be on the "outside" of the hull facet.
    # The cross product (a-b)x(b-c) points in the direction of the normal to the plane abc.
    # For a convex hull, the origin is on one side. The "empty" side is the other.
    # If the triangle is small, the circumcenter is close to the centroid.
    # Let's check the dot product with the centroid.
    centroid = (a + b + c) / 3.0
    if np.dot(center, centroid) < 0:
        center = -center
        
    cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
    r = np.arccos(cos_r)
    
    return r, center

def _compute_covering_metrics(pts):
    """
    Compute the covering radius and identify the worst facets.
    Returns:
    max_radius: float
    worst_facets: list of tuples of indices (i, j, k)
    all_radii: list of radii for all facets
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        # If hull fails, return a large radius
        return np.pi, [], []
    
    simplices = hull.simplices
    radii = []
    centers = []
    
    for tri in simplices:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        r, center = _compute_circumradius_and_center(a, b, c)
        radii.append(r)
        centers.append(center)
        
    if not radii:
        return np.pi, [], []
        
    max_radius = max(radii)
    # Find all facets that are close to the max radius
    # This helps in smoothing the gradient
    threshold = max_radius - 0.001 
    worst_indices = [i for i, r in enumerate(radii) if r > threshold]
    worst_facets = [simplices[i] for i in worst_indices]
    
    return max_radius, worst_facets, radii, centers

def _spherical_gradient_step(pts, step_size, beta=100.0):
    """
    Perform a Riemannian gradient descent step.
    The gradient is computed from the weighted circumcenters of the worst facets.
    """
    max_radius, worst_facets, radii, centers = _compute_covering_metrics(pts)
    
    if not worst_facets:
        return pts
        
    # Compute weights for each facet based on its radius
    # w_k = exp(beta * (r_k - max_radius))
    weights = np.exp(beta * (np.array(radii) - max_radius))
    
    grad = np.zeros_like(pts)
    
    # Accumulate gradient for each vertex
    for k, tri in enumerate(worst_facets):
        w = weights[k]
        if w < 1e-6:
            continue
        center = centers[k]
        for idx in tri:
            v = pts[idx]
            # Project center onto tangent space at v
            # g = center - (center . v) v
            dot_cv = np.dot(center, v)
            g = center - dot_cv * v
            grad[idx] += w * g
            
    # Normalize gradients and take step
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine(pts, deadline, max_iters=5000, initial_step=0.05, beta=50.0):
    """
    Local refinement using Riemannian gradient descent.
    """
    best_pts = pts.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _spherical_gradient_step(pts, step_size, beta)
        
        new_radius, _, _, _ = _compute_covering_metrics(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            # Slightly increase step if we are improving rapidly?
            # Or keep it constant. Let's decay slowly.
            step_size *= 0.999
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
            if no_improve_count > 50:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We use 140s for computation.
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Initial points: Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_points)
    
    # Phase 1: Refine from Fibonacci
    if time.time() < deadline - 30:
        refined_pts, refined_radius = _local_refine(points.copy(), deadline - 30, max_iters=2000, initial_step=0.1, beta=50.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 2: Multiple restarts with perturbations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial points for this restart
        if restart == 0:
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 1:
            pts = best_points + np.random.randn(N, 3) * 0.2
        elif restart == 2:
            # Random points
            pts = np.random.randn(N, 3)
        elif restart == 3:
            # Mix of best and random
            random_pts = np.random.randn(N, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
            pts = 0.5 * best_points + 0.5 * random_pts
        elif restart == 4:
            # Different Fibonacci phase
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 3
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 5:
            # Perturbed best with small noise
            pts = best_points + np.random.randn(N, 3) * 0.05
        elif restart == 6:
            # Random with structure: sort by z and twist
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            z_order = np.argsort(pts[:, 2])
            phi = np.linspace(0, 2 * np.pi * 5, N)
            pts_new = np.zeros((N, 3))
            for k, idx in enumerate(z_order):
                c, s = np.cos(phi[k]), np.sin(phi[k])
                x, y, z = pts[idx]
                pts_new[idx] = [x*c - y*s, x*s + y*c, z]
            pts = pts_new
        elif restart == 7:
            # Another random
            pts = np.random.randn(N, 3)
        elif restart == 8:
            # Best with medium noise
            pts = best_points + np.random.randn(N, 3) * 0.15
        else:
            # Best with large noise
            pts = best_points + np.random.randn(N, 3) * 0.3
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Refine this restart
        if time.time() < deadline - 5:
            refined_pts, refined_radius = _local_refine(pts, deadline - 5, max_iters=1000, initial_step=0.05, beta=50.0)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish with very small steps
    if time.time() < deadline - 2:
        refined_pts, refined_radius = _local_refine(best_points.copy(), deadline - 2, max_iters=500, initial_step=0.01, beta=100.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
