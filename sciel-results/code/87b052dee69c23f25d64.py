import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 50

def _fibonacci_spiral(n):
    """Generate n points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(points):
    """
    Compute the covering radius (max angular distance to nearest point) 
    using the convex hull facets.
    Returns the radius in radians and the hull object.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., points are coplanar or in hemisphere)
        # Use a brute force check on a grid
        # This is a safety net, shouldn't happen often with good init
        return np.pi, None

    max_radius = 0.0
    worst_facet_indices = []
    
    # Iterate over facets
    for simplex in hull.simplices:
        # Get the points forming the facet
        facet_points = points[simplex]
        
        # Compute the circumcenter of the facet on the sphere
        # The circumcenter of a spherical polygon (triangle in this case)
        # is the normalization of the cross product of the edges? 
        # No, for a triangle with vertices a, b, c on the unit sphere,
        # the circumcenter is proportional to a x b + b x c + c x a? 
        # Actually, the circumcenter of the spherical triangle is the intersection of the great circles bisecting the sides.
        # A simpler vector approach: The circumcenter c is such that |c-a| = |c-b| = |c-c| (chordal distances).
        # This implies c is in the plane of a,b,c? No.
        # The circumcenter of the *plane* triangle formed by a,b,c is not the spherical circumcenter.
        
        # Let's calculate the spherical circumradius directly.
        # For a spherical triangle with sides a,b,c (angles), the circumradius R is given by:
        # cos(R) = (cos(a) + cos(b) + cos(c) - 1) / (4 * sin(a/2) * sin(b/2) * sin(c/2)) ? No, that's not right.
        
        # Alternative: The circumradius of the facet is the angular distance from the facet's circumcenter to any vertex.
        # To find the circumcenter:
        # The circumcenter x satisfies x . a = x . b = x . c = k (for some k).
        # This means x is parallel to the vector v = a x (b - a) + b x (c - b) ... no.
        # x is in the null space of [a-b, b-c, c-a]^T?
        # Actually, x is the normalization of the vector:
        # v = (a x b) x (b x c) x (c x a) ?
        
        # Let's use a robust numerical method for the circumcenter of 3 points on a sphere.
        # The circumcenter is the point on the sphere that is equidistant from a, b, c.
        # This is the intersection of the sphere and the plane defined by the three points? No, that's the circumcenter of the planar triangle.
        # The spherical circumcenter is the pole of the great circle passing through a, b, c? No, that's only if they are on a great circle.
        
        # Correct method:
        # The circumcenter $c$ is given by:
        # $c = \frac{a \times (b \times c) + b \times (c \times a) + c \times (a \times b)}{\| \dots \|}$
        # Wait, let's derive.
        # $c \cdot a = c \cdot b = c \cdot c_{vertex}$.
        # $c \cdot (a-b) = 0$
        # $c \cdot (b-c_{vertex}) = 0$
        # So $c$ is perpendicular to $a-b$ and $b-c_{vertex}$.
        # Thus $c$ is parallel to $(a-b) \times (b-c_{vertex})$.
        
        v1 = facet_points[0] - facet_points[1]
        v2 = facet_points[1] - facet_points[2]
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-12:
            continue
            
        c_center = normal / np.linalg.norm(normal)
        
        # Ensure c_center is on the "correct" side (same side as the facet points relative to origin? 
        # The circumcenter should be close to the vertices.
        # Check dot product with one of the vertices.
        if np.dot(c_center, facet_points[0]) < 0:
            c_center = -c_center
            
        # Calculate the angular distance from center to vertex
        # This is the circumradius
        cos_r = np.clip(np.dot(c_center, facet_points[0]), -1.0, 1.0)
        r = np.arccos(cos_r)
        
        if r > max_radius:
            max_radius = r
            worst_facet_indices = [simplex]
        elif abs(r - max_radius) < 1e-6:
            worst_facet_indices.append(simplex)

    return max_radius, worst_facet_indices

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    # Initial Configuration
    points = _fibonacci_spiral(N)
    
    # Track best configuration
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)[0]
    
    # Optimization Parameters
    learning_rate = 0.05
    decay = 0.99
    repulsion_strength = 0.01
    
    # Main Optimization Loop
    while time.time() - start_time < timeout:
        current_radius, worst_facets = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if not worst_facets:
            break
            
        # Compute Gradient
        grad = np.zeros_like(points)
        
        # 1. Attraction to circumcenter of worst facets
        for simplex in worst_facets:
            facet_points = points[simplex]
            v1 = facet_points[0] - facet_points[1]
            v2 = facet_points[1] - facet_points[2]
            normal = np.cross(v1, v2)
            c_center = normal / np.linalg.norm(normal)
            if np.dot(c_center, facet_points[0]) < 0:
                c_center = -c_center
                
            # Move vertices towards circumcenter
            for idx in simplex:
                # Vector from point to center
                direction = c_center - points[idx]
                # Project direction onto tangent plane at points[idx]
                # tangent = direction - (direction . p) * p
                proj = direction - np.dot(direction, points[idx]) * points[idx]
                grad[idx] += proj
                
        # 2. Repulsion to prevent clustering and maintain spread
        # This is a heuristic to help escape local minima where facets might be small but clustered
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # Add random perturbation if coincident
                    grad[i] += np.random.normal(size=3)
                    grad[j] -= np.random.normal(size=3)
                    continue
                # Force proportional to 1/dist^2
                force = repulsion_strength / (dist * dist)
                # Project onto tangent
                grad_i = diff - np.dot(diff, points[i]) * points[i]
                grad_j = -diff - np.dot(-diff, points[j]) * points[j]
                grad[i] += force * grad_i
                grad[j] += force * grad_j
                
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
            
        # Update points
        points = points + learning_rate * grad
        
        # Normalize points back to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay learning rate
        learning_rate *= decay
        
        # Add small noise to escape local minima
        if np.random.rand() < 0.1:
            noise = np.random.normal(size=(N, 3)) * 0.01
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Final Check and Return
    # Ensure the best points are valid
    final_radius = _compute_covering_radius(best_points)[0]
    if final_radius > best_radius:
         # This shouldn't happen, but just in case
         pass
         
    return best_points
