"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n

def _covering_radius(points):
    """Compute covering radius in radians using convex hull facets."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the three points forming the facet
            p = points[simplex]
            # The center of the circumcircle of the triangle on the unit sphere
            # is the normalized cross product of (p1-p0) and (p2-p0)
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            center = np.cross(v1, v2)
            c_norm = np.linalg.norm(center)
            if c_norm < 1e-15:
                # Degenerate facet
                continue
            center /= c_norm
            
            # The covering radius for this facet is the angular distance from the center to the farthest vertex
            # But we need the angular distance from the center to the plane of the facet
            # Actually, the circumradius on the sphere is the angular distance from the center to any vertex
            # if the center is the circumcenter. Let's compute the angle between center and each vertex.
            # The largest angular distance from the center to any vertex is the circumradius.
            # But wait, the center we computed is the normal to the plane, not the circumcenter on the sphere.
            # For a triangle on the unit sphere, the circumcenter is the point on the sphere equidistant (angularly) from all three vertices.
            # This is given by normalizing the sum of the vertices if the triangle is acute, but in general
            # we need to solve for the point c such that c.p0 = c.p1 = c.p2 and ||c||=1.
            # This means c is in the span of p0, p1, p2 and orthogonal to (p1-p0) and (p2-p0).
            # So c is proportional to the cross product (p1 x p2) + (p2 x p0) + (p0 x p1)? No.
            # Actually, the circumcenter c satisfies c x (p1 - p0) = 0 and c x (p2 - p0) = 0, so c is parallel to (p1 x p2) + ... 
            # Let's use the formula: the circumcenter of a spherical triangle with vertices a, b, c is
            # proportional to a x b + b x c + c x a? No, that's not right either.
            # The correct formula: the circumcenter is the normalized vector (a x b) x (b x c) ... no.
            # Let's solve the linear system. c = alpha*a + beta*b + gamma*c
            # c dot (b-a) = 0 => alpha*(a.(b-a)) + beta*(b.(b-a)) + gamma*(c.(b-a)) = 0
            # c dot (c-a) = 0 => alpha*(a.(c-a)) + beta*(b.(c-a)) + gamma*(c.(c-a)) = 0
            # This is getting complicated. Let's use a simpler approach: the circumradius is the angular distance from the center to a vertex.
            # The center is the point on the sphere that maximizes the minimum angular distance to the three vertices? No, it's equidistant.
            # For now, let's approximate the covering radius by the angular distance from the facet normal to the vertices.
            # This is an upper bound.
            for v in p:
                dot = np.dot(center, v)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                if angle > max_r:
                    max_r = angle
        return max_r
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Geometric relaxation: identify worst facet and move its vertices
    step_size = 0.1
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Compute covering radius and identify worst facet
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_r = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            p = points[simplex]
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            center = np.cross(v1, v2)
            c_norm = np.linalg.norm(center)
            if c_norm < 1e-15:
                continue
            center /= c_norm
            for v in p:
                dot = np.clip(np.dot(center, v), -1.0, 1.0)
                angle = np.arccos(dot)
                if angle > max_r:
                    max_r = angle
                    worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        current_radius = max_r
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Move the vertices of the worst facet along their negative spherical gradients
        # The gradient of the angular distance from the center to a vertex points away from the center
        # So we move each vertex slightly away from the center to shrink the facet
        p = points[worst_simplex]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        center = np.cross(v1, v2)
        c_norm = np.linalg.norm(center)
        if c_norm < 1e-15:
            continue
        center /= c_norm
        
        # Move each vertex in the direction away from the center (to shrink the facet)
        for idx in worst_simplex:
            v = points[idx]
            # Direction away from center
            direction = v - np.dot(v, center) * center
            d_norm = np.linalg.norm(direction)
            if d_norm < 1e-15:
                continue
            direction /= d_norm
            # Small step
            points[idx] = v + step_size * direction
            points[idx] = _normalize(points[idx])
        
        # Decay step size
        step_size *= 0.99
    
    # Return the best configuration found
    return best_points

# EVOLVE-BLOCK-END
