"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds


def _normalize(pts):
    """Normalize points to unit sphere, handling zero vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point.
    
    This equals the largest circumradius of the facets of the convex hull of the points.
    """
    if len(pts) < 4:
        # Degenerate case
        return 90.0
    
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        
        for simplex in hull.simplices:
            # Get the facet points
            facet_pts = pts[simplex]
            n = len(facet_pts)
            
            # Compute the circumradius of the simplex (facet)
            # For a simplex on the unit sphere, the circumradius is the angle
            # from the origin to the circumcenter of the facet.
            
            # The circumcenter of a facet (simplex) on the unit sphere:
            # We need to find the point on the sphere that is equidistant from all facet vertices
            # and lies on the "far side" from the origin (i.e., the point that maximizes distance
            # to the facet plane).
            
            # For a simplex with vertices v_1, ..., v_k on the unit sphere,
            # the circumcenter in the affine hull of the simplex can be found.
            # The distance from origin to this circumcenter gives the covering radius for that facet.
            
            # Method: Find the point c in the affine span of the facet that is equidistant
            # from all vertices. Then the covering radius contribution is arccos(|c|/|c|) ... 
            # Actually, we need the angular distance from the sphere point closest to the facet
            # to the nearest vertex.
            
            # Better approach: The farthest point on the sphere from the facet is the point
            # along the normal from the origin to the facet plane, on the side away from the origin.
            # The angular distance from that point to any vertex of the facet is the circumradius.
            
            # Compute the normal to the facet plane
            if n == 3:
                v1, v2, v3 = facet_pts
                normal = np.cross(v2 - v1, v3 - v1)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-15:
                    continue
                normal = normal / norm_len
                
                # The point on the sphere in the direction of the normal (away from origin if needed)
                # The facet normal should point outward from the convex hull
                # Check direction: the centroid of the facet should be on the same side
                centroid = np.mean(facet_pts, axis=0)
                if np.dot(centroid, normal) < 0:
                    normal = -normal
                
                # The angular distance from the sphere point `normal` to any vertex v_i
                # is arccos(dot(normal, v_i))
                # For a proper circumcircle, all these should be equal
                cos_vals = np.dot(facet_pts, normal)
                # The circumradius is the maximum angular distance
                # Actually, for the facet, the farthest point from the facet on the sphere
                # is `normal`, and the angular distance from `normal` to each vertex is the circumradius
                # But we want the max over all vertices of the angular distance from the farthest sphere point
                # Wait, the covering radius for this facet is the angular distance from the sphere point
                # that is farthest from the facet to the nearest vertex. Since `normal` is equidistant
                # from all vertices (for a circumcircle), the distance is arccos(dot(normal, v_i)).
                
                # However, `normal` might not be exactly equidistant due to numerical issues.
                # The true circumradius is determined by the circumcircle of the triangle.
                
                # Let's compute it properly:
                # The circumcenter of triangle v1, v2, v3 in 3D (on the unit sphere)
                # is the point c = a*v1 + b*v2 + c*v3 (with a+b+c=1) such that |c-v1|=|c-v2|=|c-v3|
                # and c is in the plane of the triangle.
                
                # Actually, a simpler approach: the circumradius R satisfies
                # R = |normal| * sin(theta) where theta is the angle subtended...
                # No, let's just use the formula for circumradius of a triangle on a sphere.
                
                # For a spherical triangle with sides a, b, c (angular distances),
                # the circumradius R can be found. But this is complex.
                
                # Simpler: the point `normal` is the pole of the great circle containing the facet.
                # The angular distance from `normal` to each vertex is the same if and only if
                # the vertices lie on a small circle centered at `normal`.
                # The angular distance is arccos(dot(normal, v_i)).
                # Due to numerical precision, these may differ slightly.
                # The covering radius for this facet is the maximum of these distances.
                
                angles = np.arccos(np.clip(cos_vals, -1, 1))
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
            else:
                # Higher-dimensional simplex (shouldn't happen for 3D convex hull facets)
                continue
        
        # Convert to degrees
        return np.degrees(max_radius)
        
    except Exception:
        return 90.0


def _covering_radius_fast(pts):
    """
    Fast approximation of covering radius using the convex hull.
    """
    if len(pts) < 4:
        return 90.0
    
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            n = len(facet_pts)
            
            if n == 3:
                v1, v2, v3 = facet_pts
                # Compute normal
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-15:
                    continue
                normal = normal / norm_len
                
                # Ensure normal points outward
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(centroid, normal) < 0:
                    normal = -normal
                
                # Angular distance from normal to each vertex
                cos_vals = np.clip(np.dot(facet_pts, normal), -1, 1)
                angles = np.arccos(cos_vals)
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
        
        return np.degrees(max_radius)
        
    except Exception:
        return 90.0


def _objective(x, n=N):
    """Objective function: covering radius in degrees."""
    pts = _normalize(x.reshape(n, 3))
    return _covering_radius_fast(pts)


def _fidelity_spiral(n=N):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _greedy_refine(pts, max_iters=500, step_size=0.05):
    """
    Greedy local refinement: iteratively move each point to reduce the covering radius.
    """
    pts = _normalize(pts.copy())
    current_radius = _covering_radius_fast(pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(len(pts)):
            # Try small perturbations
            best_x = pts[i]
            best_r = current_radius
            
            for axis in range(3):
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5,
                              step_size*2, -step_size*2]:
                    test_pts = pts.copy()
                    test_pts[i, axis] += delta
                    test_pts = _normalize(test_pts)
                    r = _covering_radius_fast(test_pts)
                    if r < best_r - 1e-10:
                        best_r = r
                        best_x = test_pts[i]
            
            if not np.allclose(best_x, pts[i]):
                pts[i] = best_x
                current_radius = best_r
                improved = True
        
        if not improved:
            break
    
    return pts, current_radius


def _simulated_annealing(pts, max_iters=2000, t_start=0.5, t_end=0.01, step_scale=0.1):
    """
    Simulated annealing to escape local minima.
    """
    pts = _normalize(pts.copy())
    current_radius = _covering_radius_fast(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    n = len(pts)
    
    for it in range(max_iters):
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Choose a random point to perturb
        i = np.random.randint(n)
        delta = np.random.normal(0, step_scale, 3)
        test_pts = pts.copy()
        test_pts[i] += delta
        test_pts = _normalize(test_pts)
        
        test_radius = _covering_radius_fast(test_pts)
        delta_r = test_radius - current_radius
        
        if delta_r < 0 or np.random.rand() < np.exp(-delta_r / max(t, 1e-10)):
            pts = test_pts
            current_radius = test_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
    
    return best_pts, best_radius


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts = _fidelity_spiral(N)
    pts = _normalize(pts)
    current_radius = _covering_radius_fast(pts)
    
    best_pts = pts.copy()
    best_radius = current_radius
    
    # Greedy refinement first
    pts, current_radius = _greedy_refine(pts, max_iters=200, step_size=0.05)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return best_pts
    
    # Simulated annealing for more global exploration
    np.random.seed(42)
    pts, current_radius = _simulated_annealing(pts, max_iters=1000, t_start=0.3, t_end=0.02, step_scale=0.15)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return best_pts
    
    # Another round of greedy refinement on the SA result
    pts, current_radius = _greedy_refine(pts, max_iters=300, step_size=0.03)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return best_pts
    
    # Try a few more SA runs with different seeds
    for seed in [1, 2, 3]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        np.random.seed(seed)
        # Start from a slightly perturbed version of the best
        perturbed = best_pts + np.random.normal(0, 0.1, best_pts.shape)
        perturbed = _normalize(perturbed)
        
        pts, current_radius = _simulated_annealing(perturbed, max_iters=800, t_start=0.2, t_end=0.02, step_scale=0.12)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Quick greedy refinement
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        pts, current_radius = _greedy_refine(pts, max_iters=150, step_size=0.03)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    return best_pts
