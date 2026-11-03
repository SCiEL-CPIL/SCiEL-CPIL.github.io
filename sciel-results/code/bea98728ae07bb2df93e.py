"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object or None
        radii: array of circumradii for each facet
        max_radius: maximum circumradius
        critical_indices: indices of facets with circumradius close to max
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi, []
    
    radii = np.zeros(len(hull.simplices))
    max_radius = 0.0
    critical_indices = []
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[idx] = 0.0
            continue
            
        # Normal vector to the facet
        n = cross / norm_cross
        
        # Ensure normal points outward (away from origin)
        if np.dot(n, a) < 0:
            n = -n
            
        # Circumcenter of the triangle on the sphere
        # The circumcenter is the point on the sphere equidistant from a, b, c.
        # It lies along the normal n.
        # The distance from origin to the plane of abc is d = dot(a, n).
        # The distance from the projection of origin to a is r_plane = sqrt(1 - d^2).
        # The circumradius R (angle) satisfies sin(R) = r_plane / (distance from circumcenter to a)? No.
        # Let's use the formula: cos(R) = dot(circumcenter, a).
        # Circumcenter C = n * (1/d). Wait, C is on the sphere.
        # Actually, the circumcenter of a spherical triangle is the pole of the plane passing through a,b,c.
        # If the triangle contains the origin in its convex hull projection? No, it's a facet of the convex hull.
        # The plane equation is n . x = d.
        # The circumcenter C on the sphere is n / |n| * (1/d) ? No, C must be unit length.
        # C = n / (n . a) ?
        # Let's check: if C = n / (n . a), then C . a = 1 / (n . a) / (n . a) ? No.
        # C = n / (n . a) is not unit length unless |n|=1 and (n.a)=1.
        
        # Correct derivation:
        # Let C be the circumcenter. C is perpendicular to the plane of abc? No, C is on the sphere.
        # C is equidistant from a, b, c.
        # C lies on the line passing through the origin and the normal n? Yes, by symmetry.
        # So C = alpha * n. Since |C|=1, alpha = 1 (assuming n is unit).
        # So C = n.
        # Is n equidistant from a, b, c?
        # dot(n, a) = d. dot(n, b) = d. dot(n, c) = d.
        # So the angular distance is arccos(d) for all three.
        # Thus, the circumcenter is simply the normalized normal vector n!
        
        # Wait, is this always true?
        # Yes, for a triangle on a sphere, the circumcenter is the pole of the great circle passing through the vertices?
        # No, the circumcenter is the point equidistant from the vertices.
        # The set of points equidistant from a and b is the great circle through the poles of the great circle ab.
        # The intersection of these great circles is the pole of the plane abc.
        # So yes, C = n (or -n).
        
        # We chose n such that dot(n, a) > 0.
        # So C = n.
        # The angular distance (circumradius) is arccos(dot(n, a)).
        
        d = np.dot(n, a)
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            critical_indices = [idx]
        elif abs(r - max_radius) < 1e-6:
            critical_indices.append(idx)
            
    return hull, radii, max_radius, critical_indices

def _spherical_lloyd_iteration(pts, max_iter=50):
    """
    Perform Spherical Lloyd iterations: move each point to the centroid of its Voronoi cell.
    This is a powerful global optimizer for spherical codes.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        # Compute Voronoi cells using ConvexHull of points?
        # No, Voronoi on sphere is dual to ConvexHull.
        # The Voronoi cell of point i is the set of points on sphere closer to i than to any other.
        # The centroid of the Voronoi cell can be approximated by the average of the vertices of the dual polytope?
        # Actually, a simpler approximation: move each point towards the center of mass of its neighboring facets' circumcenters?
        # Or just use the standard Lloyd: move to the spherical centroid of the Voronoi cell.
        
        # Efficient approximation:
        # For each point i, find the facets incident to i.
        # The Voronoi cell is the convex hull of the circumcenters of those facets (on the sphere).
        # The centroid is the normalized sum of the circumcenters (weighted by area? or just unweighted).
        
        hull, radii, max_r, crit = _compute_hull_and_radii(pts)
        if hull is None:
            break
            
        # Map point index to list of incident facets
        point_to_facets = {i: [] for i in range(N)}
        for idx, simplex in enumerate(hull.simplices):
            for p_idx in simplex:
                point_to_facets[p_idx].append(idx)
                
        new_pts = np.zeros_like(pts)
        for i in range(N):
            facets = point_to_facets[i]
            if not facets:
                new_pts[i] = pts[i]
                continue
                
            # Sum of circumcenters (normals) of incident facets
            # Weight by facet area? For now, unweighted.
            centroid_vec = np.zeros(3)
            for f_idx in facets:
                simplex = hull.simplices[f_idx]
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                n = np.cross(b - a, c - a)
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    n = n / nn
                    if np.dot(n, a) < 0:
                        n = -n
                    centroid_vec += n
                    
            norm_c = np.linalg.norm(centroid_vec)
            if norm_c > 1e-12:
                new_pts[i] = centroid_vec / norm_c
            else:
                new_pts[i] = pts[i]
                
        new_pts = _to_sphere(new_pts)
        
        # Check for improvement
        old_obj = max_r
        _, _, new_obj, _ = _compute_hull_and_radii(new_pts)
        
        if new_obj < old_obj:
            pts = new_pts
        else:
            # If no improvement, break or dampen?
            # Lloyd can oscillate. If it doesn't improve, we might be stuck.
            break
            
    return pts

def _steepest_descent_critical(pts, max_iter=1000):
    """
    Steepest descent on the critical facets.
    Gradient of the covering radius w.r.t. point positions.
    """
    pts = _to_sphere(pts.copy())
    best_pts = pts.copy()
    best_obj = _compute_hull_and_radii(pts)[2]
    
    step_size = 0.05
    start_time = time.time()
    
    while time.time() - start_time < 60: # Max 60s for this phase
        hull, radii, max_r, crit_indices = _compute_hull_and_radii(pts)
        if hull is None:
            break
            
        if max_r < best_obj:
            best_obj = max_r
            best_pts = pts.copy()
            
        if not crit_indices:
            break
            
        # Compute gradient for points involved in critical facets
        grad = np.zeros_like(pts)
        
        for idx in crit_indices:
            simplex = hull.simplices[idx]
            i, j, k = simplex
            a = pts[i]
            b = pts[j]
            c = pts[k]
            
            v1 = a - b
            v2 = a - c
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            if np.dot(n, a) < 0:
                n = -n
                
            # Circumcenter is n
            cc = n
            
            # Gradient of arccos(dot(cc, a)) w.r.t. a
            # d/da arccos(x) = -1/sqrt(1-x^2) * dx/da
            # x = dot(cc, a)
            # dx/da = cc
            # So grad_a = -cc / sqrt(1 - dot(cc, a)^2)
            
            d = np.clip(np.dot(cc, a), -1.0, 1.0)
            denom = np.sqrt(1.0 - d*d)
            if denom < 1e-6:
                denom = 1e-6
                
            g_a = -cc / denom
            g_b = -cc / denom
            g_c = -cc / denom
            
            # We want to decrease the radius, so we move in the direction of -grad
            # grad points in direction of increasing radius?
            # If we move a away from cc, dot(cc, a) decreases, arccos increases.
            # So -grad points towards cc.
            
            grad[i] += g_a
            grad[j] += g_b
            grad[k] += g_c
            
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grad = grad / grad_norms
        
        # Step
        new_pts = _to_sphere(pts - step_size * grad)
        
        new_obj = _compute_hull_and_radii(new_pts)[2]
        
        if new_obj < max_r:
            pts = new_pts
            step_size *= 0.99
        else:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Phase 1: Spherical Lloyd Initialization
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _compute_hull_and_radii(pts)[2]
    
    # Run Lloyd to get a good local minimum
    lloyd_pts = _spherical_lloyd_iteration(pts, max_iter=100)
    lloyd_obj = _compute_hull_and_radii(lloyd_pts)[2]
    
    if lloyd_obj < best_obj:
        best_obj = lloyd_obj
        best_pts = lloyd_pts.copy()
        
    # Phase 2: Steepest Descent Polish
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        sd_pts, sd_obj = _steepest_descent_critical(best_pts, max_iter=1000)
        if sd_obj < best_obj:
            best_obj = sd_obj
            best_pts = sd_pts.copy()
            
    # Phase 3: Try a few random restarts with Lloyd+SD if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > _DEADLINE - 20:
            break
            
        # Perturb best points
        perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * 0.1)
        
        # Lloyd
        lloyd_pts = _spherical_lloyd_iteration(perturbed, max_iter=50)
        lloyd_obj = _compute_hull_and_radii(lloyd_pts)[2]
        
        if lloyd_obj < best_obj:
            best_obj = lloyd_obj
            best_pts = lloyd_pts.copy()
            
        # SD
        remaining = _DEADLINE - time.time()
        if remaining > 10:
            sd_pts, sd_obj = _steepest_descent_critical(best_pts, max_iter=500)
            if sd_obj < best_obj:
                best_obj = sd_obj
                best_pts = sd_pts.copy()
                
    # Final Validation
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
