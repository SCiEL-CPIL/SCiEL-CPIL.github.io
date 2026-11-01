import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_facet_radii(points):
    """
    Compute the circumradius (angular) for each facet of the convex hull.
    Returns an array of radii for each simplex in the hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.full(len(points), np.pi) # Fallback if hull fails
    
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    
    # Vectorized computation of normals and radii
    # Vertices for each facet: (n_facets, 3, 3)
    verts = points[simplices]
    
    # Calculate normals: cross(v2-v1, v3-v1)
    # Alternatively, for unit sphere, normal is proportional to p1 x p2 + p2 x p3 + p3 x p1
    # This is numerically more stable for near-coplanar points? 
    # Let's use the standard cross product of edges.
    v1, v2, v3 = verts[:, 0], verts[:, 1], verts[:, 2]
    
    # Edge vectors
    e1 = v2 - v1
    e2 = v3 - v1
    
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Avoid division by zero
    valid = norms[:, 0] > 1e-12
    normals[valid] /= norms[valid]
    
    # Ensure normals point outward (away from origin)
    # The centroid of the facet is roughly in the direction of the normal
    centroids = np.mean(verts, axis=1)
    dots = np.sum(normals * centroids, axis=1)
    # If dot < 0, flip
    flip_mask = dots < 0
    normals[flip_mask] *= -1
    
    # The circumradius of the spherical cap is the angle between the normal and any vertex.
    # Since it's a triangle on the sphere, the angle to v1, v2, v3 should be the same if the normal is the circumcenter.
    # However, the normal of the plane passing through 3 points on the sphere IS the circumcenter direction.
    # So, angle = arccos(dot(normal, v1))
    
    cos_theta = np.sum(normals * v1, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    radii[valid] = np.arccos(cos_theta[valid])
    radii[~valid] = np.pi # Degenerate facets
    
    return radii

def _gradient_of_max_radius(points, radii, top_k=5):
    """
    Compute the gradient to reduce the maximum facet radii.
    We focus on the 'top_k' largest facets to drive down the global max.
    """
    n_facets = len(radii)
    if n_facets == 0:
        return np.zeros_like(points)
    
    # Get indices of the largest radii
    # We want to minimize max(radii). A common trick is to use a soft-max or just push the top ones.
    # Here we compute the gradient for the top_k facets and sum them with weights.
    top_indices = np.argsort(radii)[-top_k:]
    
    grad = np.zeros_like(points)
    
    # Re-fetch hull to get simplices
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return grad
        
    # We need to be careful: the gradient of arccos(n . p_i) w.r.t p_i
    # Let f = arccos(n . p_i)
    # df/dp_i = -1/sqrt(1 - (n.p_i)^2) * (dn/dp_i . p_i + n . dp_i)
    # This is complex because n depends on p_i.
    
    # Alternative: Use the formula for the circumradius R of a spherical triangle with sides a,b,c.
    # Or simpler: The normal n is (p1 x p2 + p2 x p3 + p3 x p1) / |...|
    # Let S = p1 x p2 + p2 x p3 + p3 x p1. n = S / |S|.
    # Radius theta = arccos(n . p1).
    # This derivation is tedious. 
    # Instead, let's use a finite difference or a simpler heuristic: 
    # Move the vertices of the worst facets towards the centroid of their Voronoi region?
    # No, that's Lloyd.
    
    # Let's use the property that to minimize the circumradius, we should move the points 
    # in the direction that increases the minimum distance to the other points in the facet?
    # Actually, for a spherical triangle, the circumcenter is the normal.
    # If we move p1 in the direction of the normal, we decrease the angle?
    # Let's test: if p1 moves towards n, the angle decreases.
    # So, for each vertex in a "bad" facet, we can push it towards the facet normal.
    
    verts = points[simplices]
    v1, v2, v3 = verts[:, 0], verts[:, 1], verts[:, 2]
    
    # Recalculate normals for the specific facets we care about to save time? 
    # We already have normals from _compute_facet_radii, but we need to map them back.
    # It's easier to just recompute for the top_k facets.
    
    for idx in top_indices:
        s = simplices[idx]
        p1, p2, p3 = points[s[0]], points[s[1]], points[s[2]]
        
        # Normal
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12: continue
        n /= n_norm
        
        # Ensure outward
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
            
        # Push each vertex towards the normal
        # The "force" is stronger for larger radii
        weight = radii[idx]
        
        # Gradient direction for p1 to reduce angle with n is n - (n.p1)p1 (tangent component of n)
        # Actually, moving p1 towards n reduces the angle.
        # The tangent component of n at p1 is n - (n.p1)p1.
        g1 = n - np.dot(n, p1) * p1
        g2 = n - np.dot(n, p2) * p2
        g3 = n - np.dot(n, p3) * p3
        
        grad[s[0]] += weight * g1
        grad[s[1]] += weight * g2
        grad[s[2]] += weight * g3
        
    return grad

def construct():
    start_time = time.time()
    total_time_limit = 140.0 # 10s margin for 150s limit
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.pi
    
    # Optimization Loop
    # We use a simple gradient descent step on the "push towards normal" heuristic.
    # This is effectively a variant of Lloyd's but driven by the circumradius metric.
    
    step_size = 0.05
    learning_rate_decay = 0.99
    
    while time.time() - start_time < total_time_limit:
        radii = _compute_facet_radii(points)
        current_max_radius = np.max(radii)
        
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        # If we are very close to the known best (11.316 deg = 0.1975 rad), we can stop or polish
        if best_radius < 0.1976:
            break
            
        # Compute gradient
        grad = _gradient_of_max_radius(points, radii, top_k=10)
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            # Try random perturbation if stuck
            points += np.random.normal(scale=0.01, size=(N, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        grad /= grad_norm
        
        # Apply step
        points += step_size * grad
        
        # Normalize back to sphere
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= learning_rate_decay
        
        # Prevent degeneration: if points get too close, push them apart?
        # The convex hull logic handles this by creating degenerate facets with large radii,
        # which will trigger a strong gradient.
        
    return best_points

# Helper to ensure valid output
def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

# Wrap construct to be safe
def construct():
    try:
        pts = _construct_impl()
        return _normalize(pts)
    except Exception as e:
        # Fallback to Fibonacci if anything fails
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Rename the main logic to _construct_impl to keep construct() clean
# But the prompt asks to keep function names. I will just put the logic in construct() directly 
# or use a helper. The prompt says "Keep the function names and signatures of the seed program".
# The seed program has `construct()`. I will put the logic in `construct()`.

# I need to merge the above logic into `construct()` properly.

def construct():
    start_time = time.time()
    total_time_limit = 140.0
    
    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.pi
    
    step_size = 0.05
    learning_rate_decay = 0.99
    
    while time.time() - start_time < total_time_limit:
        radii = _compute_facet_radii(points)
        current_max_radius = np.max(radii)
        
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        if best_radius < 0.1976: # ~11.317 deg
            break
            
        grad = _gradient_of_max_radius(points, radii, top_k=10)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            points += np.random.normal(scale=0.01, size=(N, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        grad /= grad_norm
        
        points += step_size * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        step_size *= learning_rate_decay
        
    return best_points

