"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _calc_covering_radius(points):
    """
    Calculate the covering radius of the points on the unit sphere.
    The covering radius is the maximum circumradius of the triangular facets
    of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        # Vectorize circumradius calculation if possible, but loop is fine for N=130
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Calculate circumcenter of the triangle p1, p2, p3 in 3D space
            # The circumcenter C is equidistant from p1, p2, p3.
            # It lies on the line passing through the circumcenter of the triangle 
            # in the plane containing p1,p2,p3 and perpendicular to that plane.
            # Actually, for points on a sphere centered at origin, the circumcenter 
            # of the spherical triangle is the normalization of the vector sum 
            # weighted by sines? No, simpler:
            
            # The circumcenter of the triangle p1,p2,p3 in R^3 is the point C 
            # such that |C-p1|^2 = |C-p2|^2 = |C-p3|^2.
            # This C is collinear with the origin and the spherical circumcenter 
            # if the triangle is not degenerate? 
            # Yes, the set of points equidistant from p1, p2, p3 is a line 
            # perpendicular to the plane of the triangle. Since p1,p2,p3 are on 
            # the unit sphere, the origin is equidistant from them (distance 1).
            # So the line of equidistant points passes through the origin.
            # Thus, the circumcenter on the sphere is simply the unit vector in 
            # the direction of the vector sum? No, that's the centroid.
            
            # Let's use the formula for circumcenter of a triangle in 3D.
            # Let v = p2 - p1, w = p3 - p1.
            # The normal to the plane is n = cross(v, w).
            # The circumcenter C satisfies (C - p1) . v = |v|^2 / 2 and (C - p1) . w = |w|^2 / 2.
            # Also C = p1 + alpha * v + beta * w + gamma * n.
            # Since C is equidistant from p1, p2, p3, and the origin is also equidistant 
            # from p1, p2, p3 (all dist 1), the vector C is in the plane spanned by 
            # the normal n and the origin? 
            # Actually, the circumcenter of the spherical triangle is the unit vector 
            # in the direction of the vector:
            # C = (sin(a)*p1 + sin(b)*p2 + sin(c)*p3) / |...| ? No.
            
            # Reliable method:
            # Solve linear system for C_plane (circumcenter in the plane of p1,p2,p3)
            # Then C_sphere = normalize(C_plane) is NOT correct generally?
            # Wait, if p1,p2,p3 are on the unit sphere, the circumcenter of the 
            # Euclidean triangle p1p2p3 lies on the line connecting the origin to 
            # the spherical circumcenter? 
            # Let's check: The perpendicular bisector plane of p1p2 passes through 
            # the origin because |p1|=|p2|=1. Similarly for p2p3.
            # So the intersection of these bisector planes is a line passing through 
            # the origin. This line contains the Euclidean circumcenter of the 
            # triangle p1p2p3 and the spherical circumcenter.
            # So, we can find the Euclidean circumcenter C_eucl of the triangle 
            # p1,p2,p3, and then the spherical circumcenter is C_eucl / ||C_eucl||.
            
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            
            # Coefficients for C_eucl = p1 + t1*v + t2*w
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_eucl = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_eucl)
            if norm_C < 1e-12:
                continue
            
            C_sphere = C_eucl / norm_C
            cos_R = np.clip(np.dot(C_sphere, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def construct():
    # Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_points = points.copy()
    current_radius = _calc_covering_radius(current_points)
    
    # Simulated Annealing Parameters
    T = 0.1
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 20000 # Adjust based on time limit
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if T < T_min:
            break
            
        # Perturb a random point
        idx = rng.integers(0, N)
        # Random direction on sphere
        random_vec = rng.normal(size=3)
        random_vec /= np.linalg.norm(random_vec)
        
        # New position
        new_point = current_points[idx] + step_size * random_vec
        new_point /= np.linalg.norm(new_point)
        
        # Evaluate new radius
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        trial_radius = _calc_covering_radius(trial_points)
        
        # Acceptance criterion
        delta = trial_radius - current_radius
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_points = trial_points
            current_radius = trial_radius
        
        T *= cooling_rate
        
        # Adaptive step size
        if it % 1000 == 0 and it > 0:
             # Reduce step size as we cool
             step_size *= 0.9

    # Final local polish: Move points of worst facet towards circumcenter
    for _ in range(50):
        try:
            hull = ConvexHull(current_points)
        except Exception:
            break
        
        facet_radii = []
        for simplex in hull.simplices:
            p1, p2, p3 = current_points[simplex[0]], current_points[simplex[1]], current_points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                facet_radii.append(0.0)
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_eucl = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_eucl)
            if norm_C < 1e-12:
                facet_radii.append(0.0)
                continue
            C_sphere = C_eucl / norm_C
            cos_R = np.clip(np.dot(C_sphere, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            facet_radii.append(R)
            
        max_idx = np.argmax(facet_radii)
        if facet_radii[max_idx] < 0.197:
            break
            
        simplex = hull.simplices[max_idx]
        C_eucl = None
        p1, p2, p3 = current_points[simplex[0]], current_points[simplex[1]], current_points[simplex[2]]
        v = p2 - p1
        w = p3 - p1
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        vw = np.dot(v, w)
        det = vv * ww - vw * vw
        if abs(det) > 1e-12:
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_eucl = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_eucl)
            if norm_C > 1e-12:
                C_sphere = C_eucl / norm_C
                step = 0.005
                for idx in simplex:
                    p = current_points[idx]
                    new_p = p + step * (C_sphere - p)
                    new_p /= np.linalg.norm(new_p)
                    current_points[idx] = new_p

    # Final normalization
    current_points = current_points / np.linalg.norm(current_points, axis=1, keepdims=True)
    
    return current_points
# EVOLVE-BLOCK-END
