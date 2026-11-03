import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Helper to normalize points
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    # Helper to compute covering radius and worst facet
    def get_covering_radius_and_worst(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            worst_center = None
            
            # Vectorized calculation of normals and distances is tricky with hull.simplices
            # but we can loop over simplices. N=130, hull has ~2N simplices.
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                
                # Normal to the facet
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                
                # Ensure n points outward (away from origin). 
                # Since origin is inside, dot(n, v0) should be positive if n is outward?
                # Actually, for a convex hull of points on the sphere, the normal pointing out of the hull 
                # (towards the empty cap) is the one that has positive dot product with the vertices?
                # Let's check: The cap is the set of points y such that y dot n >= v0 dot n.
                # The center of the cap is n. The radius is arccos(n dot v0).
                # We want the cap that is "empty" of points.
                # The normal n computed by cross product might point in or out.
                # If n dot v0 is negative, the center is -n and radius is arccos(-n dot v0).
                # Actually, the circumcenter of the spherical triangle is the normalized vector 
                # perpendicular to the plane. The two candidates are n and -n.
                # The one that is "outside" the triangle (in the empty cap) is the one 
                # that is further from the origin? No, both are on the sphere.
                # The empty cap is the one that does not contain any other points.
                # For the convex hull, the facet defines a plane. The side not containing the origin 
                # is the "exterior". The normal pointing to the exterior is the center of the empty cap.
                # Since the origin is inside the convex hull, the normal pointing away from the origin 
                # is the one where dot(n, centroid) > 0?
                # Let's just compute dist for n and -n and take the one that corresponds to the empty cap.
                # Actually, for a convex hull of points on the sphere, the normal n such that 
                # n dot v0 > 0 is the outward normal if the triangle is "small".
                # Let's assume n is outward if dot(n, v0) > 0. If not, flip it.
                if n @ v0 < 0:
                    n = -n
                
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                
                if dist > max_rad:
                    max_rad = dist
                    worst_simplex = simplex
                    worst_center = n
                    
            return max_rad, worst_simplex, worst_center
        except Exception:
            return np.pi, None, None

    # Initial points: Fibonacci Spiral
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(points)

    best_points = init_fibonacci()
    best_radius, _, _ = get_covering_radius_and_worst(best_points)
    
    points = best_points.copy()
    
    step_size = 0.05
    repulsion_strength = 0.005
    
    # Pre-compute indices for repulsion to vectorize
    # We will do a simple O(N^2) loop but vectorized per pair is hard without triu.
    # Let's use a simple loop for clarity and correctness, N=130 is small.
    
    while time.time() - start_time < time_limit:
        current_radius, worst_simplex, worst_center = get_covering_radius_and_worst(points)
        
        if worst_simplex is None:
            break
            
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Move vertices of the worst facet towards the center
        if worst_center is not None:
            for idx in worst_simplex:
                points[idx] = points[idx] + step_size * (worst_center - points[idx])
        
        # Apply global repulsion to prevent clustering
        # This helps maintain a uniform distribution
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-10:
                    # If points are coincident, push them apart randomly
                    points[i] += 0.01 * np.random.randn(3)
                    points[j] -= 0.01 * np.random.randn(3)
                    continue
                dist = np.sqrt(dist_sq)
                # Force is stronger when closer
                force = repulsion_strength / dist_sq
                # Only repel if they are closer than some threshold? 
                # Global repulsion helps, but can be strong.
                # Let's use a soft repulsion.
                points[i] += force * (diff / dist)
                points[j] -= force * (diff / dist)
                
        points = normalize(points)
        
        # Adaptive step size
        # If we improved, keep or increase slightly?
        # If we degraded, decrease.
        # We don't have the "previous" radius easily here without storing it.
        # Let's just slowly decay the step size.
        step_size *= 0.995
        repulsion_strength *= 0.999
        
        if step_size < 1e-6:
            break
            
    return best_points
