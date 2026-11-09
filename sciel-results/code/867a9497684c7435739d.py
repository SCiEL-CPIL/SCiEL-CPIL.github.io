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

def _compute_face_data(pts):
    """
    Compute circumradius, center, and vertex indices for all faces of the convex hull.
    Returns:
    radii: array of radii
    centers: array of centers
    simplices: list of vertex indices for each face
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return None, None, None
        
    simplices = hull.simplices
    radii = np.zeros(len(simplices))
    centers = np.zeros((len(simplices), 3))
    
    for idx, tri in enumerate(simplices):
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Compute circumcenter and radius
        # Cross product of (a-b) and (b-c) gives normal direction
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-10:
            # Degenerate case
            center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-10:
                radii[idx] = np.pi
                centers[idx] = np.array([0,0,0])
                continue
            center = center / norm_c
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
        else:
            center = cross / norm_cross
            # Ensure correct orientation (dot with centroid > 0)
            centroid = (a + b + c) / 3.0
            if np.dot(center, centroid) < 0:
                center = -center
            
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
            
    return radii, centers, simplices

def _compute_covering_radius(pts):
    radii, _, _ = _compute_face_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _riemannian_steepest_descent(pts, step_size, beta=10.0):
    """
    Riemannian steepest descent on the sphere for the covering radius.
    
    The key insight: for the worst facet (largest circumradius), we want to
    move its vertices toward the circumcenter. On the sphere, the direction
    that reduces the circumradius is the projection of (center - vertex) 
    onto the tangent space at each vertex.
    
    We use a soft-max weighting to smoothly blend contributions from multiple
    near-worst facets, avoiding the nonsmoothness of the max operation.
    """
    radii, centers, simplices = _compute_face_data(pts)
    if radii is None:
        return pts
        
    # Soft-max weights: emphasize worst facets but include near-worst ones
    max_r = np.max(radii)
    # Scale beta so that facets within ~1/radian of max get significant weight
    weights = np.exp(beta * (radii - max_r))
    weights /= np.sum(weights)
    
    # Build gradient in ambient space, then project to tangent space
    grad = np.zeros_like(pts)
    
    for k, tri in enumerate(simplices):
        w = weights[k]
        if w < 1e-10:
            continue
            
        center = centers[k]
        
        for idx in tri:
            v = pts[idx]
            # Direction from vertex to circumcenter
            diff = center - v
            # Project onto tangent space at v: remove component along v
            dot_v = np.dot(diff, v)
            tangent = diff - dot_v * v
            grad[idx] += w * tangent
            
    # Normalize and apply step on the sphere
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            # Riemannian step: exp map on sphere
            # For small steps, v + step * g is sufficient, then renormalize
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _dual_uniformization_step(pts, step_size, beta=10.0):
    """
    Perform a step that uniformizes the circumradii of the faces.
    This treats the problem as flattening the 'height' field of the dual polytope.
    """
    radii, centers, simplices = _compute_face_data(pts)
    if radii is None:
        return pts
        
    max_r = np.max(radii)
    
    # Weights based on deviation from max radius
    # Higher weight for faces that are "bad" (large radius)
    # Using exponential weighting to focus on the worst but include others
    weights = np.exp(beta * (radii - max_r))
    
    # Initialize gradient accumulator
    grad = np.zeros_like(pts)
    
    # Accumulate "pull" towards circumcenters for each vertex
    # The idea: if a face has a large radius, its circumcenter is "far" from the vertices.
    # Moving vertices towards the circumcenter of the bad faces helps reduce the radius?
    # Actually, moving vertices towards the circumcenter of the face they belong to
    # tends to make the face more "equilateral" or adjust its size.
    # Let's project the circumcenter onto the tangent space of the vertex.
    
    for k, tri in enumerate(simplices):
        w = weights[k]
        if w < 1e-8:
            continue
            
        center = centers[k]
        
        for idx in tri:
            v = pts[idx]
            # Project center onto tangent plane at v
            dot_cv = np.dot(center, v)
            # g is the component of center perpendicular to v
            g = center - dot_cv * v
            grad[idx] += w * g
            
    # Apply step
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine_dual(pts, deadline, max_iters=2000, initial_step=0.05, beta=10.0):
    """
    Local refinement using dual uniformization.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _dual_uniformization_step(pts, step_size, beta)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.995
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6 or no_improve_count > 100:
                break
                
    return best_pts, best_radius

def _local_refine_riemannian(pts, deadline, max_iters=2000, initial_step=0.03, beta=15.0):
    """
    Local refinement using Riemannian steepest descent on the sphere.
    
    This is a more precise, geometry-driven approach that directly targets
    the worst facet by moving its vertices toward the circumcenter.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _riemannian_steepest_descent(pts, step_size, beta)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.995
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6 or no_improve_count > 100:
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
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Dual Uniformization Refinement
    if time.time() < deadline - 30:
        refined_pts, refined_radius = _local_refine_dual(points.copy(), deadline - 30, max_iters=3000, initial_step=0.05, beta=10.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 2: Riemannian steepest descent from best point
    if time.time() < deadline - 25:
        refined_pts, refined_radius = _local_refine_riemannian(best_points.copy(), deadline - 25, max_iters=2000, initial_step=0.03, beta=15.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 3: Multiple restarts with different initializations
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            pts = best_points + np.random.randn(N, 3) * 0.05
        elif restart == 1:
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 2:
            # Random
            pts = np.random.randn(N, 3)
        elif restart == 3:
            # Fibonacci with phase shift
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 4
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 4:
            # Mix
            random_pts = np.random.randn(N, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
            pts = 0.8 * best_points + 0.2 * random_pts
        else:
            # Perturbed best
            pts = best_points + np.random.randn(N, 3) * (0.05 + 0.05 * restart)
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Refine using both methods: dual first, then Riemannian
        if time.time() < deadline - 8:
            refined_pts, refined_radius = _local_refine_dual(pts, deadline - 8, max_iters=800, initial_step=0.05, beta=10.0)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
        if time.time() < deadline - 5:
            refined_pts, refined_radius = _local_refine_riemannian(best_points.copy(), deadline - 5, max_iters=1000, initial_step=0.03, beta=15.0)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish with Riemannian
    if time.time() < deadline - 2:
        refined_pts, refined_radius = _local_refine_riemannian(best_points.copy(), deadline - 2, max_iters=1000, initial_step=0.01, beta=20.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
