import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # If hull fails (e.g., points in hemisphere), return a large penalty
        return np.pi
        
    max_radius = 0.0
    for simplex in simplices:
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            max_radius = max(max_radius, np.pi / 2)
            continue
            
        u = n / norm_n
        
        # Ensure u points outward (away from origin)
        # The centroid of the triangle is (a+b+c)/3. 
        # The normal should point in the direction of the centroid from the origin?
        # Actually, for a convex hull of points on the unit sphere, the facet normal 
        # points away from the origin.
        # Let's check dot product with centroid. If negative, flip.
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # The circumradius of the spherical triangle is the angle 
        # between the facet normal u and the vertices.
        # All vertices are at the same angular distance from the normal if the triangle is equilateral?
        # No, the circumcenter of the spherical triangle is the point on the sphere 
        # farthest from the triangle's interior? 
        # Actually, the covering radius for a facet is the distance from the 
        # "pole" of the facet (u) to the vertices. 
        # The pole is the point on the sphere closest to the plane of the facet?
        # No, the pole is the intersection of the line perpendicular to the facet 
        # through the origin with the sphere.
        # The angular distance from the pole u to vertex a is arccos(dot(u, a)).
        # This distance is the same for all vertices of the facet ONLY IF the facet 
        # is equilateral? No.
        # Wait. The circumradius of a spherical triangle is the radius of the 
        # circumscribed circle. The center of this circle is the pole u?
        # Yes, the pole u is the center of the spherical circle passing through a, b, c.
        # Therefore, the distance from u to a, u to b, and u to c are all equal 
        # to the circumradius.
        # Let's verify. 
        # dot(u, a) = dot(u, b) = dot(u, c) ?
        # u is normal to the plane containing a, b, c.
        # So a, b, c lie on a plane. The distance from the origin to the plane is h.
        # u is the unit normal. 
        # dot(u, a) is the projection of a onto u. Since a is on the unit sphere, 
        # and u is the normal to the plane a-b-c, the projection is the same for 
        # all points on the plane. 
        # Yes, dot(u, a) == dot(u, b) == dot(u, c).
        # So we just need to compute arccos(dot(u, a)).
        
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radius = np.arccos(cos_theta)
        
        if radius > max_radius:
            max_radius = radius
            
    return max_radius

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Initial points: Fibonacci sphere
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    best_points = points.copy()
    best_radius = compute_covering_radius(points)
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-5
    cooling = 0.99
    step_size = 0.1
    
    current_radius = best_radius
    current_points = points.copy()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        
        # Generate a random perturbation direction
        # Perturb the point on the sphere
        # We can add a small random vector and re-normalize
        # Or rotate the point slightly
        # Let's add a random vector
        noise = np.random.randn(3) * step_size
        new_point = current_points[idx] + noise
        norm = np.linalg.norm(new_point)
        if norm < 1e-6:
            continue
        new_point /= norm
        
        # Temporarily update
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        # Evaluate new radius
        new_radius = compute_covering_radius(current_points)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Reject
            current_points[idx] = old_point
            
        # Cool down
        T *= cooling
        
    return best_points
