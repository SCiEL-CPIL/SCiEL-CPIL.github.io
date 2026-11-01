<<<<<<< SEARCH
import numpy as np
import scipy.spatial
import time

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians and the index of the worst point (circumcenter).
    """
    # Convex hull of the points
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., points are coplanar or degenerate)
        # Calculate max distance to nearest point on a grid
        grid = _generate_grid(100)
        dists = np.array([np.min(np.arccos(np.clip(np.dot(grid, p), -1, 1))) for p in points])
        return np.max(dists), -1

    worst_radius = 0
    worst_facet_idx = -1
    worst_circumcenter = None

    # Iterate over facets (triangles) of the convex hull
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Normal to the plane
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # The circumcenter is either n or -n.
        # We choose the one that is "close" to the triangle (dot product > 0 with a)
        if np.dot(n, a) < 0:
            n = -n
            
        # Distance from circumcenter to vertices
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_facet_idx = simplex[0] # Just an index, we need the simplex
            # Store simplex indices
            worst_simplex = simplex
            worst_circumcenter = n

    return worst_radius, worst_simplex, worst_circumcenter

def _generate_grid(n_theta=50, n_phi=100):
    """Generate a grid of points on the sphere for fallback evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _local_search(points, max_steps=500, step_size=0.1):
    """
    Local search to minimize the covering radius.
    Moves the vertices of the worst facet towards its circumcenter.
    """
    current_points = points.copy()
    best_radius, _, _ = _covering_radius(current_points)
    
    for step in range(max_steps):
        radius, simplex, circumcenter = _covering_radius(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # If radius is very small, we might be stuck or converged
        if radius < 1e-6:
            break
            
        # Get the 3 points of the worst facet
        idx1, idx2, idx3 = simplex
        
        # Move these points towards the circumcenter
        # We use a small step size to avoid overshooting
        alpha = step_size
        
        p1 = current_points[idx1]
        p2 = current_points[idx2]
        p3 = current_points[idx3]
        
        # Move towards circumcenter
        # new_p = normalize(p + alpha * (circumcenter - p))
        # This is a gradient step on the distance to the circumcenter
        
        # To be more robust, we can move all 3 points
        new_p1 = p1 + alpha * (circumcenter - p1)
        new_p2 = p2 + alpha * (circumcenter - p2)
        new_p3 = p3 + alpha * (circumcenter - p3)
        
        # Normalize
        new_p1 = new_p1 / np.linalg.norm(new_p1)
        new_p2 = new_p2 / np.linalg.norm(new_p2)
        new_p3 = new_p3 / np.linalg.norm(new_p3)
        
        current_points[idx1] = new_p1
        current_points[idx2] = new_p2
        current_points[idx3] = new_p3
        
        # Occasionally, we might want to perturb other points to escape local minima
        # But for now, this greedy approach is fast.
        
    return best_points, best_radius
=======
import numpy as np
import scipy.spatial
import time

N = 50

def _compute_covering_radius_and_grads(points):
    """
    Compute the covering radius and gradients for points on the unit sphere.
    Returns:
    radius: float, the covering radius in radians
    grads: array of shape (N, 3), the gradient of the radius w.r.t points
    worst_simplex: array of indices of the worst facet
    circumcenter: array of shape (3,), the circumcenter of the worst facet
    """
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        return np.pi, np.zeros_like(points), np.array([0, 0, 0]), np.zeros(3)
    points = points / norms
    
    # Compute Convex Hull
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: return large radius and zero grads to signal failure
        return np.pi, np.zeros_like(points), np.array([0, 0, 0]), np.zeros(3)

    worst_radius = 0
    worst_simplex = None
    worst_circumcenter = None

    # Iterate over facets (triangles) of the convex hull
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Normal to the plane
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # The circumcenter is either n or -n.
        # We choose the one that is "close" to the triangle (dot product > 0 with a)
        if np.dot(n, a) < 0:
            n = -n
            
        # Distance from circumcenter to vertices
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_simplex = simplex
            worst_circumcenter = n

    # Compute gradients
    grads = np.zeros_like(points)
    if worst_simplex is not None and worst_circumcenter is not None:
        c = worst_circumcenter
        # Gradient of arccos(c.p) w.r.t p is -(c - (c.p)p) / sqrt(1-(c.p)^2)
        # Here c.p = cos(R) for the vertices of the worst facet
        
        # We only apply gradient to the vertices of the worst simplex
        # because the radius is determined by this facet locally.
        idxs = worst_simplex
        for idx in idxs:
            p = points[idx]
            c_p = np.dot(c, p)
            # Avoid division by zero if c_p is close to +/- 1
            s_r = np.sqrt(1 - c_p**2 + 1e-12)
            # grad = - (c - c_p * p) / s_r
            grads[idx] = -(c - c_p * p) / s_r

    return worst_radius, grads, worst_simplex, worst_circumcenter

def _generate_grid(n_theta=50, n_phi=100):
    """Generate a grid of points on the sphere for fallback evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _local_search(points, max_steps=1000, step_size=0.01, time_limit=None):
    """
    Local search using gradient descent on the covering radius.
    """
    current_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius_and_grads(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time_limit and time.time() > time_limit:
            break
            
        radius, grads, simplex, circumcenter = _compute_covering_radius_and_grads(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # If radius is very small or we are stuck, break
        if radius < 1e-6:
            break
            
        # Gradient step
        # Project gradient onto tangent space
        dot_grads_p = np.sum(grads * current_points, axis=1, keepdims=True)
        tangent_grads = grads - dot_grads_p * current_points
        
        # Step
        new_points = current_points - step_size * tangent_grads
        
        # Normalize
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_points = new_points
        
    return best_points, best_radius
>>>>>>> REPLACE

