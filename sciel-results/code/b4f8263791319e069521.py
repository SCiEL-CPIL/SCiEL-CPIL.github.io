import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    # Deadline guard: 120s limit, aim for 110s
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

    # Helper to calculate covering radius from Delaunay triangulation
    def calc_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                # Compute circumcenter of triangle v1, v2, v3
                # Formula for circumcenter in 3D for a triangle
                # a = v2 - v1, b = v3 - v1
                # c = a x b
                # d = 2 * dot(v1, c)
                # circumcenter = (dot(v1, v1) * (v2 x v3) + dot(v2, v2) * (v3 x v1) + dot(v3, v3) * (v1 x v2)) / (2 * dot(v1, c))
                # Simpler: Project to 2D or use formula
                a = v2 - v1
                b = v3 - v1
                c = np.cross(a, b)
                denom = 2.0 * np.dot(v1, c)
                if np.abs(denom) < 1e-10:
                    # Degenerate triangle
                    rad = np.linalg.norm(v1 - v2)
                else:
                    # Circumcenter formula
                    # C = (|v1|^2 (v2 x v3) + |v2|^2 (v3 x v1) + |v3|^2 (v1 x v2)) / (2 (v1 x v2) x (v1 x v3)) ? No.
                    # Let's use the standard formula for circumcenter of 3 points
                    # D = 2 * (v1x * (v2y * v3z - v2z * v3y) - v1y * (v2x * v3z - v2z * v3x) + v1z * (v2x * v3y - v2y * v3x))
                    # This is 2 * dot(v1, cross(v2, v3)) if origin is 0? No.
                    
                    # Easier: Solve linear system for center C such that |C-v1|^2 = |C-v2|^2 = |C-v3|^2
                    # 2*C dot (v2-v1) = |v2|^2 - |v1|^2
                    # 2*C dot (v3-v1) = |v3|^2 - |v1|^2
                    # C is on the plane through v1, v2, v3? No, C is on the normal line.
                    # Since it's on the sphere, C is the normalized normal? No, circumcenter of the triangle lies on the sphere only if the triangle is right? No.
                    # The circumcenter of the spherical triangle is the intersection of the great circle arcs.
                    # In 3D, it is the point on the sphere equidistant from v1, v2, v3.
                    # It lies on the line passing through the origin and the vector c = (v1 x v2) x (v2 x v3)?
                    # Actually, the circumcenter of a triangle on the unit sphere is simply the normalized vector:
                    # c = (v1 + v2 + v3) / |v1 + v2 + v3| ? No, that's the centroid.
                    
                    # Correct method:
                    # The circumcenter C satisfies C dot v1 = C dot v2 = C dot v3 = cos(theta)
                    # This implies C is parallel to (v1 x v2) x (v2 x v3) ?
                    # Let's use the formula:
                    # C = (v1 * |v2 x v3|^2 + v2 * |v3 x v1|^2 + v3 * |v1 x v2|^2) / |v1 x v2 x v3|^2 ? No.
                    
                    # Let's just use the 2D projection method or solve the linear system.
                    # System:
                    # 2 * C dot a = |v2|^2 - |v1|^2 = 0 (since on unit sphere)
                    # 2 * C dot b = |v3|^2 - |v1|^2 = 0
                    # So C is orthogonal to a and b? That means C is parallel to c = a x b.
                    # Wait, if C is parallel to a x b, then C is the normal to the plane of the triangle.
                    # Does the normal to the plane of the triangle lie on the sphere?
                    # Yes, for a triangle inscribed in a sphere, the circumcenter is the intersection of the perpendicular bisectors.
                    # The vector (v1 x v2) x (v2 x v3) is the circumcenter direction?
                    # Actually, for points on the unit sphere, the circumcenter is:
                    # C = normalize( v1 + v2 + v3 ) is NOT correct generally.
                    
                    # Let's derive:
                    # |C - v1|^2 = |C|^2 + |v1|^2 - 2 C dot v1 = 2 - 2 C dot v1 (since |C|=1, |v1|=1)
                    # We want to minimize/maximize this.
                    # The circumcenter C satisfies C dot v1 = C dot v2 = C dot v3.
                    # Let C = k * (v1 x v2) x (v2 x x v3) ?
                    # Actually, the vector n = (v1 x v2) x (v2 x v3) is the circumcenter vector.
                    # Let's verify.
                    # n dot v1 = (v1 x v2) dot v1 * (v2 x v3) ... no.
                    
                    # Simpler:
                    # C is proportional to (v1 + v2 + v3) ONLY if the triangle is equilateral.
                    
                    # Let's use the linear algebra solution.
                    # C lies in the span of v1, v2, v3? No.
                    # C is a linear combination of v1, v2, v3?
                    # Since C dot (v2-v1) = 0 and C dot (v3-v1) = 0, C is orthogonal to the plane of the triangle.
                    # So C is parallel to the normal vector n = (v2-v1) x (v3-v1).
                    # So C = normalize(n).
                    # Let's check if this C is equidistant.
                    # C dot v1 = normalize(n) dot v1.
                    # C dot v2 = normalize(n) dot v2.
                    # Since n is orthogonal to v2-v1, n dot (v2-v1) = 0 => n dot v2 = n dot v1.
                    # So yes, the circumcenter of a spherical triangle is the normalized normal vector to the plane of the triangle.
                    
                    n = np.cross(a, b)
                    norm_n = np.linalg.norm(n)
                    if norm_n < 1e-10:
                        rad = 0.0 # Degenerate
                    else:
                        C = n / norm_n
                        # Distance is angle between C and v1
                        cos_theta = np.dot(C, v1)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        rad = np.arccos(cos_theta)
                if rad > max_rad:
                    max_rad = rad
            return max_rad
        except:
            return np.pi # Invalid hull, return large value

    current_radius = calc_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()

    # Local Optimization Loop
    # We will perform gradient-like steps.
    # Step size schedule: Start large, decrease.
    step_size = 0.05
    iterations = 0
    max_iterations = 500
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        try:
            hull = ConvexHull(points)
        except:
            break
            
        # Identify critical triangles
        # For each triangle, compute circumcenter and radius
        # Move vertices of large-radius triangles towards the circumcenter
        
        updates = np.zeros_like(points)
        
        for simplex in hull.simplices:
            v1_idx, v2_idx, v3_idx = simplex
            v1, v2, v3 = points[v1_idx], points[v2_idx], points[v3_idx]
            
            a = v2 - v1
            b = v3 - v1
            n = np.cross(a, b)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-10:
                continue
                
            C = n / norm_n
            cos_theta = np.dot(C, v1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            rad = np.arccos(cos_theta)
            
            # If this radius is significant, apply a force
            # Force direction: Move v towards C
            # Strength proportional to rad? Or just constant for critical ones?
            # Let's apply a small move towards C for all, weighted by rad
            weight = rad / (np.pi/2) # Normalize roughly
            
            # Move v1 towards C
            updates[v1_idx] += (C - v1) * weight
            updates[v2_idx] += (C - v2) * weight
            updates[v3_idx] += (C - v3) * weight
        
        # Normalize updates and apply
        # We need to ensure points stay on the sphere
        # new_points = points + step_size * updates
        # Then normalize
        
        # Check if updates are zero
        if np.allclose(updates, 0):
            break
            
        new_points = points + step_size * updates
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        new_radius = calc_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
            # Slow down step size slightly if improving
            step_size *= 0.99
        else:
            # If not improving, reduce step size and try again
            step_size *= 0.9
            if step_size < 0.001:
                break
                
    # Final check
    final_radius = calc_covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius

    # Convert to degrees for score? No, the function returns points.
    # The evaluator calculates the score.
    return best_points

# EVOLVE-BLOCK-END
