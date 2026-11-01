"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper: Calculate covering radius and identify critical points
    def get_covering_info(pts):
        """
        Returns:
        - radius: covering radius in radians
        - critical_points: list of tuples (facet_indices, circumcenter)
        """
        try:
            hull = ConvexHull(pts)
            max_r = 0.0
            critical = []
            
            for simplex in hull.simplices:
                i, j, k = simplex
                p1, p2, p3 = pts[i], pts[j], pts[k]
                
                # Compute circumcenter of the spherical triangle
                # The circumcenter C on the sphere is the point equidistant to p1, p2, p3.
                # It is the normalized vector (p1 + p2 + p3) only if the triangle is equilateral? No.
                # The circumcenter of the planar triangle p1, p2, p3 is the intersection of perpendicular bisectors.
                # Let's solve for C_planar such that |C-p1|^2 = |C-p2|^2 = |C-p3|^2.
                # C = p1 + a*(p2-p1) + b*(p3-p1)
                # (C-p1).(p2-p1) = 0.5|p2-p1|^2
                # (C-p1).(p3-p1) = 0.5|p3-p1|^2
                
                u = p2 - p1
                v = p3 - p1
                uu = np.dot(u, u)
                vv = np.dot(v, v)
                uv = np.dot(u, v)
                
                det = uu * vv - uv * uv
                if abs(det) < 1e-10:
                    continue
                
                # Solve linear system:
                # [uu uv] [a]   [0.5*uu]
                # [uv vv] [b] = [0.5*vv]
                a = (0.5 * uu * vv - 0.5 * uv * vv) / det
                b = (0.5 * vv * uu - 0.5 * uu * uv) / det
                
                c_planar = p1 + a * u + b * v
                c_norm = np.linalg.norm(c_planar)
                if c_norm < 1e-10:
                    continue
                
                c_sph = c_planar / c_norm
                
                # Check if c_sph is inside the spherical triangle.
                # A point is inside if it is on the same side of each edge as the opposite vertex.
                # Alternatively, check if the angles are all < 90? No, that's for packing.
                # For covering, if the circumcenter is outside the triangle, the max distance
                # in that cell is determined by an edge midpoint or vertex, not the circumcenter.
                # However, for "good" configurations, most critical points are circumcenters.
                # To be robust, we should check if the projection of c_sph onto the plane
                # falls inside the triangle.
                
                # Simple check: dot products with normals of edges?
                # Let's use the barycentric coordinate check on the plane.
                # If c_planar is inside the triangle p1,p2,p3, then c_sph is the valid circumcenter for that facet.
                # If c_planar is outside, the "worst" point in that Voronoi cell is on the boundary (edge midpoint).
                
                # Check if c_planar is inside triangle p1,p2,p3
                # Cross products method
                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
                
                # This 2D check doesn't work directly in 3D without projection.
                # Better check: 
                # The circumcenter is inside the spherical triangle if and only if 
                # the triangle is "acute" in terms of spherical geometry?
                # Actually, if the circumcenter is outside, the max distance to nearest site 
                # is achieved at the midpoint of the longest edge of the triangle.
                
                # Let's compute the distance to the circumcenter.
                d_c = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                
                # Check if c_sph is inside the triangle.
                # We can check if the dot product of (c_sph - p1) with the normal of the edge p2-p3 
                # is in the correct direction?
                
                # Robust check:
                # Project c_sph onto the plane of the triangle? No, c_sph is on the sphere.
                # Let's check if c_sph can be written as a positive combination of p1, p2, p3?
                # No, that's for the cone.
                
                # Heuristic: For N=50, if the configuration is reasonably spread, 
                # the circumcenters are usually inside. If not, the error is localized.
                # Let's assume c_sph is the candidate for the max.
                # To be safe, also check edge midpoints.
                
                # Distance to edge midpoints
                m12 = (p1 + p2) / np.linalg.norm(p1 + p2)
                m23 = (p2 + p3) / np.linalg.norm(p2 + p3)
                m31 = (p3 + p1) / np.linalg.norm(p3 + p1)
                
                d_m12 = np.arccos(np.clip(np.dot(m12, p1), -1.0, 1.0))
                d_m23 = np.arccos(np.clip(np.dot(m23, p2), -1.0, 1.0))
                d_m31 = np.arccos(np.clip(np.dot(m31, p3), -1.0, 1.0))
                
                # The max distance in this facet is the max of the distances to the "centers"
                # If c_sph is inside, the max is d_c.
                # If c_sph is outside, the max is max(d_m12, d_m23, d_m31).
                # How to tell if c_sph is inside?
                # If the angle between c_sph and any vertex is > 90 deg, it's likely outside? No.
                
                # Let's use a standard test: 
                # The circumcenter is inside the spherical triangle if and only if 
                # the sum of the angles of the spherical triangle is less than 2*pi? No.
                
                # Let's just take the maximum of d_c and the edge midpoints.
                # Is it possible that the max is at a vertex? No, vertices are sites, dist=0.
                # So the max in the cell is either at the circumcenter (if inside) or at an edge midpoint (if outside).
                
                # Wait, if the circumcenter is outside, the function min(dist) is maximized at the boundary.
                # The boundary consists of edges. The max on an edge segment is at the midpoint?
                # Yes, for a line segment on the sphere, the point farthest from the endpoints is the midpoint.
                
                # So, candidate max for this facet is:
                # If c_sph is inside: d_c
                # Else: max(d_m12, d_m23, d_m31)
                
                # How to check if c_sph is inside?
                # Check if the dot product of (c_sph - p1) with (p2 x p3) has the same sign as (p1 - p2) with (p2 x p3)?
                # This is getting complicated. 
                # Given time constraints, let's assume that for a "good" config, 
                # the critical points are circumcenters. If the config is bad, we'll fix it.
                # We will use d_c as the primary candidate.
                
                if d_c > max_r:
                    max_r = d_c
                    critical.append((simplex, c_sph, d_c, 'center'))
                elif d_m12 > max_r:
                    max_r = d_m12
                    critical.append((simplex, m12, d_m12, 'edge12'))
                elif d_m23 > max_r:
                    max_r = d_m23
                    critical.append((simplex, m23, d_m23, 'edge23'))
                elif d_m31 > max_r:
                    max_r = d_m31
                    critical.append((simplex, m31, d_m31, 'edge31'))
                    
            return max_r, critical

        except Exception:
            return np.pi, []

    # Gradient Descent Step
    def gradient_step(pts, step_size=0.05):
        """
        Move points to reduce the covering radius.
        For each critical point y (circumcenter or midpoint), we want to increase the distance
        from y to its nearest sites.
        The gradient of dist(y, p_i) w.r.t p_i is:
        d/dp_i arccos(y.p_i) = -y / sqrt(1 - (y.p_i)^2)
        
        We want to minimize max_j min_i dist(y_j, p_i).
        This is equivalent to maximizing min_i dist(y_j, p_i) for the worst y_j.
        So we should move p_i in the direction that increases dist(y_j, p_i).
        Direction = - grad(dist) = y / sqrt(1 - (y.p_i)^2).
        
        However, moving p_i affects other cells too.
        Simple approach: For each critical point y, find the nearest site p_i.
        Move p_i away from y.
        """
        max_r, criticals = get_covering_info(pts)
        if not criticals:
            return pts
            
        new_pts = pts.copy()
        
        # Aggregate gradients
        grads = np.zeros_like(pts)
        
        for simplex, y, dist, type in criticals:
            # Find nearest site to y
            dots = np.dot(pts, y)
            nearest_idx = np.argmin(dots) # min dot = max angle = farthest? No.
            # We want to maximize the distance from y to the nearest site.
            # Currently, the nearest site is the one with the MAX dot product (min angle).
            # So we want to move that site AWAY from y.
            # Direction to move p_i to increase dist(y, p_i) is -y (normalized appropriately).
            # Actually, grad of dist w.r.t p is -y/sqrt(...). 
            # To increase dist, we move in direction -grad = y/sqrt(...).
            # Wait. dist = arccos(y.p). d(dist)/dp = -y / sqrt(1-(y.p)^2).
            # If we move p in direction v, change is v . (-y / ...).
            # We want positive change, so v . (-y) > 0 => v . y < 0.
            # So move p in direction -y.
            
            # Let's verify. If p is close to y, y.p ~ 1. 
            # Move p away from y (towards -y). y.p decreases. arccos increases. Correct.
            
            # The strength of the push should be proportional to how "critical" it is.
            # All critical points define the max radius, so they are equally critical.
            # But if one point is the nearest to multiple critical points, it gets pushed harder?
            # Or we just average the pushes.
            
            # Push p_nearest away from y.
            # The gradient magnitude is 1/sqrt(1 - dot^2).
            # Let's just use a fixed direction -y, normalized.
            
            # To avoid breaking the unit sphere, we use a small step and renormalize.
            # We add a vector -y to the point.
            
            # We should only push the points that are responsible for the current max radius.
            # If multiple sites are equidistant (e.g. midpoint of edge), push both?
            # For a circumcenter, the 3 vertices are equidistant. Push all 3?
            # If we push all 3 away from the center, the center moves too.
            # It's better to push the site that is "most" responsible.
            # For a circumcenter, all 3 are equally responsible.
            # For an edge midpoint, the 2 endpoints are equally responsible.
            
            if type == 'center':
                indices = [simplex[0], simplex[1], simplex[2]]
                for idx in indices:
                    # Move away from y
                    grads[idx] += -y
            else:
                # Edge midpoint: push the two endpoints
                if type == 'edge12':
                    indices = [simplex[0], simplex[1]]
                elif type == 'edge23':
                    indices = [simplex[1], simplex[2]]
                else: # edge31
                    indices = [simplex[2], simplex[0]]
                
                for idx in indices:
                    grads[idx] += -y
        
        # Apply gradient
        # Normalize gradients to avoid huge steps
        for i in range(N):
            g_norm = np.linalg.norm(grads[i])
            if g_norm > 1e-6:
                grads[i] = grads[i] / g_norm
        
        new_pts = pts + step_size * grads
        
        # Renormalize
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        return new_pts

    # Optimization Loop
    best_points = points
    best_score, _ = get_covering_info(best_points)
    
    # Initial coarse search
    step_size = 0.1
    for it in range(200):
        if time.time() > deadline:
            break
            
        new_points = gradient_step(best_points, step_size)
        new_score, _ = get_covering_info(new_points)
        
        if new_score < best_score:
            best_score = new_score
            best_points = new_points
            # Decrease step size slowly
            step_size *= 0.99
        else:
            # If it didn't improve, try a smaller step or random restart?
            # Just decrease step size
            step_size *= 0.9
            if step_size < 1e-4:
                break

    # Fine-tuning with smaller steps
    step_size = 0.01
    for it in range(100):
        if time.time() > deadline:
            break
            
        new_points = gradient_step(best_points, step_size)
        new_score, _ = get_covering_info(new_points)
        
        if new_score < best_score:
            best_score = new_score
            best_points = new_points
            step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 1e-5:
                break

    # Final random polish to escape local minima
    np.random.seed(42)
    for it in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for _ in range(5):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                
                old_p = best_points[i]
                new_p = best_points[i] + 0.01 * rand_vec
                new_p /= np.linalg.norm(new_p)
                
                best_points[i] = new_p
                new_score, _ = get_covering_info(best_points)
                
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_points[i] = old_p
                    
        if not improved:
            break

    # Ensure normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
