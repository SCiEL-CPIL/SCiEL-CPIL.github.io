import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on a unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _covering_radius(points):
    """
    Compute the covering radius in radians using the convex hull.
    The covering radius is the maximum over all facets of the angular distance
    from the facet's circumcenter (on the sphere) to its vertices.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback: worst case (hemisphere)
    
    max_angle = 0.0
    # Iterate over all simplices (facets) of the convex hull
    for simplex in hull.simplices:
        # Extract vertices
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the normal to the plane containing the triangle
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        # Ensure the normal points outward (away from origin)
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        # The circumcenter on the unit sphere is in the direction of `normal`
        # (This is true because the plane of the triangle is at some distance from origin,
        # and the sphere's center is the origin. The point on the sphere closest to the plane
        # is along the normal. For a spherical triangle, the circumcenter is indeed the
        # intersection of the normal line with the sphere, provided the triangle is "small" enough
        # or generally, the pole of the plane. Wait, the circumcenter of a spherical triangle
        # is the intersection of the perpendicular bisectors of the sides.
        # For three points on a sphere, the circumcenter is the point C on the sphere such that
        # |C-v1| = |C-v2| = |C-v3|.
        # This implies C is equidistant from v1, v2, v3.
        # The set of points equidistant from v1 and v2 is the plane perpendicular to v2-v1 passing through their midpoint.
        # The intersection of these planes is the line passing through the origin and the vector v1+v2+v3?
        # Actually, for points on the unit sphere, the circumcenter is in the direction of the vector
        # n = (v1+v2) x (v2+v3) ... no.
        # Let's stick to the standard geometric fact: The circumcenter of a triangle inscribed in a sphere
        # lies on the line passing through the center of the sphere and the circumcenter of the planar triangle.
        # The circumcenter of the planar triangle (v1,v2,v3) is C_planar.
        # The line is O-C_planar. The point on the sphere is C_sphere = C_planar / ||C_planar||.
        # Is C_planar in the direction of the normal?
        # Yes, the circumcenter of a triangle lies on the line perpendicular to the plane passing through the centroid? No.
        # The circumcenter lies on the perpendicular bisectors.
        # The vector normal to the plane is n.
        # The circumcenter of the planar triangle is not generally in the direction of the normal to the plane from the origin?
        # Wait, the plane is defined by v1, v2, v3. The normal is n.
        # The circumcenter of the planar triangle lies in the plane.
        # The circumcenter of the spherical triangle is the point on the sphere in the direction of the vector
        # that is the solution to the system.
        # Actually, a simpler property: The circumcenter of the spherical triangle is the point on the sphere
        # that maximizes the minimum distance to the vertices? No, that's the center of the largest empty cap.
        # The circumcenter is the center of the cap passing through the 3 vertices.
        # This center is the point C such that C.v1 = C.v2 = C.v3.
        # This implies C is in the null space of the matrix formed by (v1-v2) and (v1-v3)?
        # No, C . (v1-v2) = 0 and C . (v1-v3) = 0.
        # So C is parallel to (v1-v2) x (v1-v3).
        # This is exactly the normal to the plane containing v1, v2, v3!
        # So yes, the circumcenter on the sphere is in the direction of the cross product (v2-v1) x (v3-v1).
        
        # Calculate angular distance from circumcenter (normal) to vertices
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        # The radius of the cap is the max of these distances
        facet_radius = max(d1, d2, d3)
        
        # Check if this cap is empty.
        # A cap is empty if no other point is closer to the circumcenter than the vertices.
        # Distance from circumcenter to any point v is arccos(C . v).
        # We need to ensure min_{k not in simplex} arccos(C . v_k) >= facet_radius.
        # Equivalent to max_{k not in simplex} C . v_k <= cos(facet_radius).
        # Since cos is decreasing, this is correct.
        
        # For efficiency, we can check this. If it's not empty, this facet doesn't define the covering radius
        # because the "worst point" for this facet is covered by another point.
        # However, the covering radius is the max over ALL points on the sphere.
        # The worst point is always the circumcenter of some facet of the convex hull.
        # If the cap is not empty, the circumcenter is not the farthest point from the set?
        # Actually, if the cap is not empty, there is a point v_k inside the cap.
        # Then the distance from the circumcenter to v_k is smaller than the distance to the vertices.
        # But is the circumcenter the farthest point from the *nearest* configuration point?
        # The covering radius is max_{y in S} min_{x in X} d(y,x).
        # The function f(y) = min_{x in X} d(y,x) is convex on the sphere (locally).
        # Its maximum occurs at a point where the gradient is zero (if interior) or on the boundary of a Voronoi cell.
        # The Voronoi cells are defined by the convex hull facets.
        # The center of the Voronoi cell is the circumcenter of the facet.
        # If the circumcenter is outside the Voronoi cell (i.e., cap is not empty), then the maximum of the min-distance
        # in that region is not at the circumcenter.
        # Wait, if the circumcenter is not in the Voronoi cell, it means there is a point v_k such that d(C, v_k) < d(C, v_i).
        # Then C is not the nearest point to v_k? No, C is the center.
        # If C is not in the Voronoi cell of v_i, it means C is closer to some other v_k.
        # Then the distance from C to its nearest neighbor is d(C, v_k).
        # Is d(C, v_k) < d(C, v_i)? Yes.
        # So the value of the objective function at C is smaller than the circumradius of the facet.
        # Does this mean we can ignore this facet?
        # Yes, if the circumcenter is not the nearest point to the facet vertices, then the circumcenter is not the "worst" point for that facet's region.
        # Actually, the maximum of the min-distance function over the sphere is attained at a point that is the circumcenter of some facet, AND that circumcenter is in the Voronoi cell of the facet's vertices.
        # If it's not in the Voronoi cell, the maximum is elsewhere.
        # So we only need to consider facets where the circumcenter is in the Voronoi cell (i.e., the cap is empty).
        
        # Optimization: Check emptiness
        # dots = points @ normal
        # min_dot = np.min(dots) # This is the max angular distance? No, min dot is max angle.
        # Wait, we want to check if any point is closer to normal than the vertices.
        # Closer means larger dot product.
        # So we check if max_{k not in simplex} (points[k] . normal) > cos(facet_radius).
        # If yes, the cap is not empty.
        
        # For N=130, checking all points for each facet is O(N^2) which is fine (130^2 = 16900).
        dots = points @ normal
        # Mask out the vertices
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        max_dot_other = np.max(dots[mask])
        
        # If max_dot_other > cos(facet_radius), then there is a point closer than the vertices.
        # In that case, the circumcenter is not the local maximum of the min-distance function.
        # So we skip this facet.
        if max_dot_other > np.cos(facet_radius) + 1e-12:
            continue
            
        max_angle = max(max_angle, facet_radius)
        
    return max_angle

def _repulsion_optimize(points, max_iters=1000, lr=0.01, deadline=None):
    """
    Optimize points on sphere using repulsion from nearest neighbors.
    This is a heuristic to spread points out.
    """
    points = points.copy().astype(float)
    n = len(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise dot products
        dots = np.clip(points @ points.T, -1, 1)
        angles = np.arccos(dots)
        np.fill_diagonal(angles, 0)
        
        # Repulsion: push points away from nearby points
        # Use a kernel that decays with distance
        # Force is sum of vectors away from neighbors, weighted by distance
        # Only consider neighbors within a certain angle to save computation?
        # For N=130, full matrix is fine.
        
        # We want to maximize the minimum distance.
        # A common heuristic is to move points in the direction of the gradient of the sum of repulsive potentials.
        # Potential V = sum_{i<j} f(d_ij).
        # To maximize min distance, we can minimize sum of 1/d_ij or similar.
        # Let's use a simple force: f_i = sum_{j != i} (p_i - p_j) / (1 + d_ij^2)
        # This pushes p_i away from p_j.
        
        # Vectorized force calculation
        # p_i - p_j is (points[:, None, :] - points[None, :, :])
        # We want to sum over j for each i.
        
        # To avoid O(N^2) memory if N was large, but N=130 is small.
        diff = points[:, None, :] - points[None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)
        
        # Weight: 1 / (1 + dist_sq)
        # Set diagonal to 0
        weights = 1.0 / (1.0 + dist_sq)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j weights[i,j] * diff[i,j]
        forces = np.sum(weights[:, :, None] * diff, axis=1)  # (N, 3)
        
        # Project force onto tangent plane
        # radial = (forces . p_i) * p_i
        radial = (forces * points).sum(axis=1, keepdims=True) * points
        tangent_forces = forces - radial
        
        # Update
        points = points + lr * tangent_forces
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial configurations
    inits = []
    # Fibonacci
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configurations
    for pts in inits:
        r = _covering_radius(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 10:
            break
            
    # Optimize from best initial config
    if best_points is not None and time.time() < deadline - 5:
        opt_pts = _repulsion_optimize(best_points, max_iters=2000, lr=0.005, deadline=deadline - 5)
        r = _covering_radius(opt_pts)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
            
    # Try optimizing from other initial configs if time permits
    for pts in inits:
        if time.time() > deadline - 10:
            break
        # Skip if it's the one we already optimized
        if np.allclose(pts, best_points):
            continue
            
        opt_pts = _repulsion_optimize(pts, max_iters=1000, lr=0.005, deadline=deadline - 5)
        r = _covering_radius(opt_pts)
        if r < best_radius:
            best_radius = r
            best_points = opt_pts.copy()
            
    # Final refinement: Local search / Hill climbing on the best solution
    # We can do a few steps of gradient descent on the covering radius directly?
    # Or just more repulsion steps with smaller lr?
    if best_points is not None and time.time() < deadline - 2:
        # Fine-tune with smaller learning rate
        opt_pts = _repulsion_optimize(best_points, max_iters=500, lr=0.001, deadline=deadline - 1)
        r = _covering_radius(opt_pts)
        if r < best_radius:
            best_points = opt_pts.copy()
            
    # Ensure output is valid
    if best_points is None:
        best_points = inits[0]
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
