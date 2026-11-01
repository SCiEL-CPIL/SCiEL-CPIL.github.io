import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_data(pts):
    """
    Compute convex hull and extract facet data:
    - For each facet (triangle), compute the circumcenter (pole) and circumradius.
    - Return: poles (F,3), radii (F,), face_indices (F,3)
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        n_faces = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute edge vectors
        AB = B - A
        AC = C - A
        
        # Cross product gives normal direction
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate facets
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return None, None, None
            
        A_v = A[valid]
        B_v = B[valid]
        C_v = C[valid]
        cross_v = cross[valid]
        norms_v = norms[valid]
        faces_v = faces[valid]
        
        # Compute circumcenter for each triangle
        # Using the formula: circumcenter = A + (|AC|^2 * (B-A) x (n) + |AB|^2 * (n) x (C-A)) / (2 * |cross|^2)
        # Actually, let's use a more robust method.
        # For a triangle on the unit sphere, the circumcenter (pole) is the normalized cross product of two edge vectors.
        # But that's only for right triangles. For general triangles, we need the actual circumcenter.
        
        # Method: The circumcenter of triangle ABC satisfies:
        # |X - A|^2 = |X - B|^2 = |X - C|^2
        # This gives 2 linear equations in X (3 unknowns), so we solve for X in the plane of the triangle.
        
        # Let's use the vector formula for circumcenter:
        # X = A + (|AC|^2 * (AB x n) + |AB|^2 * (n x AC)) / (2 * |cross|^2)
        # where n = cross / |cross|
        
        n = cross_v / (norms_v + _EPS)
        
        AB_v = B_v - A_v
        AC_v = C_v - A_v
        
        AB_sq = np.sum(AB_v * AB_v, axis=1, keepdims=True)
        AC_sq = np.sum(AC_v * AC_v, axis=1, keepdims=True)
        
        # AB x n and n x AC
        AB_x_n = np.cross(AB_v, n)
        n_x_AC = np.cross(n, AC_v)
        
        numerator = AC_sq * AB_x_n + AB_sq * n_x_AC
        denominator = 2.0 * (norms_v + _EPS)
        
        circumcenters = A_v + numerator / denominator
        
        # Normalize circumcenters to unit sphere (they should already be close)
        cc_norms = np.linalg.norm(circumcenters, axis=1, keepdims=True)
        poles = circumcenters / (cc_norms + _EPS)
        
        # Ensure outward direction
        centroids = (A_v + B_v + C_v) / 3.0
        signs = np.sum(poles * centroids, axis=1, keepdims=True)
        poles = poles * np.sign(signs)
        
        # Compute circumradii: angular distance from pole to any vertex
        cos_theta = np.sum(poles * A_v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return poles, radii, faces_v
        
    except Exception:
        return None, None, None

def _hull_radius(pts):
    """Compute the true covering radius using the convex hull."""
    poles, radii, faces = _hull_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _gradient_descent_step(pts, learning_rate, momentum=0.9):
    """
    Perform one gradient descent step using facet circumcenter data.
    For each vertex, compute the average circumcenter of its incident facets,
    weighted by the circumradius of each facet (larger radius = more important).
    Move the vertex toward this weighted average.
    """
    poles, radii, faces = _hull_data(pts)
    if poles is None:
        return pts
    
    n_faces = len(poles)
    if n_faces == 0:
        return pts
    
    # For each vertex, accumulate the weighted circumcenter direction
    grad = np.zeros_like(pts)
    
    for f in range(n_faces):
        face = faces[f]
        pole = poles[f]
        r = radii[f]
        
        # Weight by radius (larger circumradius facets are worse, so weight more)
        w = r
        
        for v in face:
            # The gradient direction for vertex v is (pole - pts[v])
            # This moves the vertex toward the circumcenter, reducing the circumradius
            grad[v] += w * (pole - pts[v])
    
    # Normalize gradient to unit sphere
    grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    valid = grad_norms.flatten() > _EPS
    grad[valid] /= grad_norms[valid]
    
    # Update points
    new_pts = pts + learning_rate * grad
    
    # Normalize to unit sphere
    new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / (new_pts_norms + _EPS)
    
    return new_pts

def _local_polish(pts, deadline):
    """
    Perform local polishing with adaptive learning rate.
    """
    lr = 0.1
    best_pts = pts.copy()
    best_r = _hull_radius(best_pts)
    
    while time.time() < deadline - 5.0:
        new_pts = _gradient_descent_step(best_pts, lr)
        new_r = _hull_radius(new_pts)
        
        if new_r < best_r:
            best_pts = new_pts
            best_r = new_r
            # Slowly decrease learning rate
            lr *= 0.95
        else:
            # If no improvement, try a smaller step or perturb
            lr *= 0.8
            if lr < 1e-6:
                break
    
    return best_pts

def construct():
    deadline = time.time() + 145.0
    
    # Start with Fibonacci
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Multi-restart with different initializations
    rng = np.random.default_rng(42)
    n_restarts = 8
    
    # Allocate time for restarts and final polish
    restart_time = (deadline - 20.0) / (n_restarts + 1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = rng.standard_normal((N, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Run gradient descent for this restart
        start_t = time.time()
        lr = 0.15
        max_steps = 200
        
        for step in range(max_steps):
            if time.time() > start_t + restart_time * 0.9:
                break
            
            pts = _gradient_descent_step(pts, lr)
            curr_r = _hull_radius(pts)
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
            
            # Adapt learning rate
            if step % 10 == 0:
                lr *= 0.98
    
    # Final polish on the best configuration
    if time.time() < deadline - 10.0:
        best_pts = _local_polish(best_pts, deadline)
        final_r = _hull_radius(best_pts)
        if final_r < best_r:
            best_r = final_r
    
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
