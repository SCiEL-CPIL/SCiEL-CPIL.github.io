"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n

def _covering_radius(points):
    """
    Compute the covering radius (in degrees) of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configured point. This equals the maximum circumradius of the
    spherical triangles (facets) of the spherical Voronoi diagram, which corresponds
    to the circumradius of the facets of the convex hull of the points.
    """
    points = np.array(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest neighbor approach
        # This is expensive but robust
        # Sample many points and find max distance to nearest
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2*np.pi, 400)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        sample_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # For each sample point, find nearest configured point
        # Use broadcasting for moderate sizes
        # dist^2 = 2 - 2*dot
        dots = sample_pts @ points.T  # shape (n_samples, N)
        min_dots = dots.max(axis=1)  # max dot = min angular distance
        max_angle = np.arccos(np.clip(1 - min_dots, -1, 1))
        return np.degrees(max_angle.max())
    
    # Get the facets (triangles) of the convex hull
    # For each facet, compute the circumradius of the spherical triangle
    # The circumradius of a spherical triangle with vertices a, b, c (unit vectors)
    # can be computed from the dihedral angles or directly.
    
    # A point on the sphere is maximally far from all configured points at the 
    # circumcenter of a spherical triangle facet. The covering radius is the max
    # over all facets of the distance from the facet's circumcenter to its vertices.
    
    # For a spherical triangle with vertices A, B, C (unit vectors), the circumcenter
    # is in the direction of A + B + C (for small triangles) or more precisely,
    # the point equidistant from A, B, C on the sphere.
    
    # The circumradius R satisfies: for the circumcenter O, 
    # cos(R) = O · A = O · B = O · C
    # where O is the unit circumcenter.
    
    # The circumcenter of a spherical triangle can be found as follows:
    # It's the intersection of the sphere with the plane perpendicular to... 
    # Actually, for a triangle on a sphere, the circumcenter is the unit vector
    # in the direction of the cross products.
    
    # A simpler approach: the circumcenter O satisfies O·A = O·B = O·C.
    # This means O is orthogonal to (A-B) and (A-C), so O is parallel to (A-B)×(A-C).
    # But this is the normal to the plane containing A, B, C, which is the center
    # of the great circle, not the spherical circumcenter in general.
    
    # Actually for a spherical triangle, the circumcenter is indeed the unit vector
    # parallel to (B-A)×(C-A) only if the triangle is "small". Let me think again.
    
    # The correct approach: The circumcenter O of spherical triangle ABC satisfies
    # O·A = O·B = O·C = cos(R)
    # So O is a unit vector such that O·(A-B) = 0 and O·(A-C) = 0.
    # This means O is parallel to (A-B)×(A-C).
    # Let n = (A-B)×(A-C), then O = ±n/|n|.
    # We choose the sign so that O·A > 0 (circumcenter on the same side as the triangle).
    # Then cos(R) = O·A.
    
    max_cos_R = 1.0  # minimum radius corresponds to max cos
    # Actually we want MAX radius, so MIN cos(R)
    min_cos_R = 1.0
    
    for simplex in hull.simplices:
        A = points[simplex[0]]
        B = points[simplex[1]]
        C = points[simplex[2]]
        
        # Normal to the plane containing A, B, C
        n = np.cross(B - A, C - A)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        O = n / nn
        
        # Choose sign so that O points toward the triangle (O·A should be positive for acute-ish)
        if O.dot(A) < 0:
            O = -O
        
        cos_R = O.dot(A)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        # The circumradius R = arccos(cos_R)
        # We want the maximum R, so minimum cos_R
        if cos_R < min_cos_R:
            min_cos_R = cos_R
    
    # Also need to check: is the circumcenter actually inside the spherical triangle?
    # If not, the farthest point in that facet's cap is at a vertex or edge midpoint.
    # For now, let's also check edge midpoints and vertices as candidates.
    
    # Actually, the covering radius is the maximum over all points y of min_i angle(y, p_i).
    # The maximum occurs either at a circumcenter of a spherical triangle (if it's inside
    # the triangle) or at a midpoint of an edge of the Voronoi diagram.
    
    # Let me compute more carefully. For each facet, compute the circumcenter and
    # check if it's within the spherical triangle. If yes, the covering radius candidate
    # is the circumradius. If not, the farthest point is on the boundary.
    
    # A robust approach: for each facet, the farthest point from all points in the 
    # Voronoi cell is either the circumcenter (if inside) or a vertex of the cell.
    
    # Let me use a different approach: directly compute the covering radius by
    # checking all facet circumcenters and edge midpoints.
    
    # Recompute: for each triangular facet, the circumcenter candidate
    results = []
    for simplex in hull.simplices:
        A = points[simplex[0]]
        B = points[simplex[1]]
        C = points[simplex[2]]
        
        n = np.cross(B - A, C - A)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        O = n / nn
        if O.dot(A) < 0:
            O = -O
        
        # Check if O is "inside" the spherical triangle ABC.
        # O is inside if it can be written as a positive combination of A, B, C
        # (in the sense of the spherical convex hull).
        # A simpler check: O should have positive dot product with each of the
        # "inward" normals of the edges.
        
        # For edge AB, the inward normal (toward C) is cross(A, B) normalized,
        # and C should be on the positive side.
        nAB = np.cross(A, B)
        nAB_norm = np.linalg.norm(nAB)
        if nAB_norm < 1e-15:
            continue
        nAB = nAB / nAB_norm
        if nAB.dot(C) < 0:
            nAB = -nAB
        if O.dot(nAB) < -1e-10:
            # O is outside across edge AB
            # The farthest point is on edge AB, at the midpoint of the great circle arc
            # Actually, it's the point on the great circle through A and B that is
            # equidistant from A and B and farthest from C.
            # For simplicity, use the midpoint of the arc AB.
            mid = A + B
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-15:
                mid = mid / mid_norm
                # The distance from mid to A
                cos_d = mid.dot(A)
                cos_d = np.clip(cos_d, -1.0, 1.0)
                results.append(np.arccos(cos_d))
            continue
        
        nBC = np.cross(B, C)
        nBC_norm = np.linalg.norm(nBC)
        if nBC_norm < 1e-15:
            continue
        nBC = nBC / nBC_norm
        if nBC.dot(A) < 0:
            nBC = -nBC
        if O.dot(nBC) < -1e-10:
            mid = B + C
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-15:
                mid = mid / mid_norm
                cos_d = mid.dot(B)
                cos_d = np.clip(cos_d, -1.0, 1.0)
                results.append(np.arccos(cos_d))
            continue
        
        nCA = np.cross(C, A)
        nCA_norm = np.linalg.norm(nCA)
        if nCA_norm < 1e-15:
            continue
        nCA = nCA / nCA_norm
        if nCA.dot(B) < 0:
            nCA = -nCA
        if O.dot(nCA) < -1e-10:
            mid = C + A
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-15:
                mid = mid / mid_norm
                cos_d = mid.dot(C)
                cos_d = np.clip(cos_d, -1.0, 1.0)
                results.append(np.arccos(cos_d))
            continue
        
        # O is inside the triangle, use circumradius
        cos_R = O.dot(A)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        results.append(np.arccos(cos_R))
    
    if len(results) == 0:
        # Fallback
        return 90.0
    
    return np.degrees(max(results))


def _gradient_descent_optimize(points, max_iters=5000, lr=0.01, deadline=None):
    """
    Optimize points on the sphere using gradient descent on the covering radius.
    Uses a smooth approximation via the log-sum-exp of squared distances.
    """
    points = np.array(points, dtype=float)
    N = len(points)
    
    # Represent points as unit vectors, parameterize by their coordinates
    # but project back to sphere after each step.
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute the covering radius and identify the worst facet
        # For gradient computation, use a smooth surrogate:
        # f(points) = log(sum_i exp(alpha * d_i^2)) / alpha
        # where d_i are angular distances from the worst point to each configured point.
        
        # Actually, let's use a simpler approach: 
        # The objective is max_y min_i angle(y, p_i).
        # We can approximate this by sampling and using a smooth max.
        
        # Alternative: use the fact that the covering radius is determined by
        # the facets. For each facet, the circumradius is a smooth function of
        # the three points defining it (when the circumcenter is inside).
        
        # Let's use a different approach: repulsive force model.
        # Each point is pushed away from its neighbors to maximize the minimum
        # distance, which approximates minimizing the covering radius.
        
        # Compute pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, 1.0)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        
        # The covering radius is related to the maximum of the minimum distances.
        # Use a soft-max of the negative minimum distances.
        
        # For each point i, the "local covering radius" contribution is the
        # distance to the nearest other point divided by 2 (roughly).
        # Better: use the facet-based approach.
        
        # Let's use gradient descent on a smooth surrogate.
        # Surrogate: sum over all pairs of soft-min of angles, or 
        # use the log-sum-exp of exp(-beta * angle) for each pair.
        
        # Actually, a cleaner approach: minimize the sum of exp(beta * r_i) 
        # where r_i is the distance from point i to its nearest neighbor.
        # This encourages uniform distribution.
        
        min_angles = angles.min(axis=1)  # nearest neighbor angle for each point
        
        # Gradient of min_angles w.r.t. points is complex.
        # Let's use a repulsive force approach instead.
        
        # Force on each point: sum over j of repulsive force from j
        # This is like a Thomson problem but with a different potential.
        
        # Use a Riesz s-energy with s=1 (Coulomb-like) but on the sphere.
        # Actually, for covering, we want to maximize the minimum distance,
        # which is related to the Riesz energy with s -> infinity.
        
        # Let's use a practical approach: 
        # 1. Compute the current covering radius
        # 2. Find the worst facet (largest circumradius)
        # 3. Move the three points of that facet slightly outward from the circumcenter
        
        # This is a local search approach.
        
        # Compute hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_circ_r = 0
        worst_facet = None
        worst_circ_center = None
        
        for simplex in hull.simplices:
            A = points[simplex[0]]
            B = points[simplex[1]]
            C = points[simplex[2]]
            
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                continue
            O = n / nn
            if O.dot(A) < 0:
                O = -O
            
            cos_R = np.clip(O.dot(A), -1.0, 1.0)
            circ_r = np.arccos(cos_R)
            
            if circ_r > max_circ_r:
                max_circ_r = circ_r
                worst_facet = simplex
                worst_circ_center = O
        
        if worst_facet is None:
            break
        
        # Move the three points of the worst facet slightly away from the circumcenter
        # to reduce the circumradius.
        # The circumradius decreases if we move the points toward the circumcenter? No,
        # we want to increase the minimum distance, so we should move points away
        # from each other, i.e., toward the circumcenter would make the triangle
        # smaller, which might help or hurt.
        
        # Actually, to reduce the circumradius of a spherical triangle, we can
        # move the vertices toward the circumcenter (making the triangle smaller)
        # or move them apart (making the triangle larger).
        
        # The covering radius is the circumradius, so we want to DECREASE it.
        # Moving vertices toward the circumcenter decreases the circumradius.
        
        # But we also need to not increase other circumradii.
        
        # Let's try moving each of the three points slightly toward the circumcenter.
        step_size = 0.005
        
        new_points = points.copy()
        for idx in worst_facet:
            p = new_points[idx]
            # Move toward circumcenter
            direction = worst_circ_center - p
            direction = direction / (np.linalg.norm(direction) + 1e-15)
            new_points[idx] = p + step_size * direction
            # Renormalize
            new_points[idx] = new_points[idx] / (np.linalg.norm(new_points[idx]) + 1e-15)
        
        # Check if this improves the covering radius
        new_radius = _covering_radius(new_points)
        if new_radius < best_radius - 1e-10:
            points = new_points
            best_points = points.copy()
            best_radius = new_radius
        else:
            # Try moving away from circumcenter
            new_points = points.copy()
            for idx in worst_facet:
                p = new_points[idx]
                direction = p - worst_circ_center
                direction = direction / (np.linalg.norm(direction) + 1e-15)
                new_points[idx] = p + step_size * direction
                new_points[idx] = new_points[idx] / (np.linalg.norm(new_points[idx]) + 1e-15)
            
            new_radius = _covering_radius(new_points)
            if new_radius < best_radius - 1e-10:
                points = new_points
                best_points = points.copy()
                best_radius = new_radius
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Local optimization using facet-based local search
    if time.time() < deadline - 30:
        points, radius = _gradient_descent_optimize(points, max_iters=3000, deadline=deadline - 10)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # Try multiple random restarts if time permits
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random initial configuration
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Optimize
        if time.time() < deadline - 15:
            opt_pts, opt_radius = _gradient_descent_optimize(pts, max_iters=2000, deadline=deadline - 10)
            if opt_radius < best_radius:
                best_points = opt_pts
                best_radius = opt_radius
    
    # Final check
    if time.time() > deadline - 5:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
