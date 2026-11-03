"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    The covering radius is the maximum, over all points y on the sphere, of the
    angular distance from y to the nearest point in pts.
    
    This equals the maximum circumradius of the facets of the convex hull of pts,
    where each facet is checked to have an empty cap (i.e., the facet is on the
    boundary of the convex hull).
    
    Returns the covering radius in radians.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute via dense sampling
        return _covering_radius_fallback(pts)
    
    # For each facet, compute the circumradius
    # The circumradius of a facet (triangle) inscribed in a sphere of radius 1
    # is related to the angles. For a triangle with vertices on the unit sphere,
    # the circumradius R satisfies: the angular distance from the center of the
    # circumscribed circle (on the sphere) to each vertex is R.
    
    # Actually, the covering radius is the maximum over all points on the sphere
    # of the distance to the nearest point. This is equivalent to finding the
    # largest empty circle on the sphere.
    
    # The vertices of the largest empty circle are either:
    # 1. A single point (if it's the antipode of another point)
    # 2. Two antipodal points
    # 3. Three points forming a triangle
    
    # Using the convex hull approach:
    # For each facet of the convex hull, the center of the circumscribed sphere
    # of that facet (in 3D) projected onto the unit sphere gives a candidate
    # for the center of a largest empty circle.
    
    # Actually, let's use a more direct approach:
    # The covering radius is the maximum, over all points on the sphere, of the
    # minimum distance to any point in pts.
    
    # This can be found by checking:
    # 1. The circumcenters of all triangles formed by triples of points
    # 2. The antipodes of all points
    # 3. The midpoints of all pairs of antipodal points (if any)
    
    # But this is O(N^3) which is too slow. Let's use the convex hull facets.
    
    # For each facet of the convex hull, the circumcenter of the facet (in 3D)
    # lies on the line through the origin and the circumcenter. The point on the
    # unit sphere closest to this circumcenter (in the direction away from the
    # origin) is a candidate for the center of a largest empty circle.
    
    # Actually, the correct approach is:
    # The covering radius is the maximum, over all points y on the sphere, of
    # min_i ||y - x_i|| (angular distance).
    
    # This is equivalent to the radius of the largest empty circle on the sphere.
    # The center of such a circle is either:
    # - The circumcenter of a triangle formed by 3 points (on the unit sphere)
    # - The antipode of a point
    # - The midpoint of 2 antipodal points
    
    # For a triangle with vertices a, b, c on the unit sphere, the circumcenter
    # of the triangle (in 3D) is the point equidistant from a, b, c. The angular
    # distance from this circumcenter to each vertex is the circumradius.
    
    # Let me compute this properly.
    
    # For each facet, get the 3 vertices
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute the circumcenter of the triangle abc on the unit sphere
        # The circumcenter is the point on the unit sphere equidistant from a, b, c
        # This is the normalized vector in the plane of the triangle that is
        # equidistant from all three vertices.
        
        # The circumcenter of a triangle on the unit sphere can be found by:
        # It lies in the direction of a + b + c (for acute triangles) or
        # we need to solve for the point x on the unit sphere such that
        # x·a = x·b = x·c
        
        # x·a = x·b => x·(a-b) = 0
        # x·a = x·c => x·(a-c) = 0
        # x·x = 1
        
        # So x is orthogonal to (a-b) and (a-c), which means x is parallel to
        # (a-b) × (a-c)
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            # Degenerate triangle
            continue
        
        x = cross / norm_cross
        
        # Check if this is the correct direction (should be on the same side as the origin)
        # The circumcenter should be such that the angular distance to each vertex
        # is minimized. We need to check both x and -x.
        
        # The angular distance from x to a is arccos(x·a)
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        # Check the other direction
        dot_a_neg = np.dot(-x, a)
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        # The circumradius is the minimum of the two (the one closer to the vertices)
        # Actually, we want the center that is farthest from all vertices in the
        # "empty" region. Let me think again.
        
        # The covering radius is the maximum over all y of min_i d(y, x_i).
        # For a given facet, the point on the sphere that is farthest from all
        # points in the facet is the circumcenter of the facet.
        
        # The circumcenter of a triangle on the unit sphere:
        # If the triangle is acute, the circumcenter is inside the triangle.
        # The angular distance from the circumcenter to each vertex is the circumradius.
        
        # Let's compute both possibilities and take the one that gives a valid
        # circumcenter (i.e., the point is on the "outer" side of the triangle).
        
        # For the circumcenter to be valid, it should be such that the cap
        # centered at x with radius r1 contains no other points.
        
        # Let's just compute the circumradius for both x and -x and take the max.
        r_pos = r1
        r_neg = r2
        
        # The covering radius candidate is the larger of the two circumradii
        # (since we want the maximum over all points on the sphere)
        candidate = max(r_pos, r_neg)
        max_radius = max(max_radius, candidate)
    
    # Also check antipodes
    for i in range(len(pts)):
        # The antipode of pts[i] is -pts[i]
        # The distance from -pts[i] to pts[i] is pi
        # But we need to check if -pts[i] is closer to some other point
        # The covering radius at -pts[i] is min_j d(-pts[i], pts[j])
        # = min_j (pi - d(pts[i], pts[j])) for j != i
        # = pi - max_j d(pts[i], pts[j]) for j != i
        
        # Actually, let's just compute the distance from -pts[i] to all points
        neg_pt = -pts[i]
        dists = np.arccos(np.clip(pts @ neg_pt, -1.0, 1.0))
        min_dist = np.min(dists)
        max_radius = max(max_radius, min_dist)
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via dense sampling."""
    # Use a fine grid on the sphere
    n_theta = 200
    n_phi = 400
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        for phi in phis:
            y = np.array([st*np.cos(phi), st*np.sin(phi), ct])
            dists = np.arccos(np.clip(pts @ y, -1.0, 1.0))
            min_dist = np.min(dists)
            if min_dist > max_radius:
                max_radius = min_dist
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _local_optimize(pts, max_time=None):
    """
    Local optimization using gradient-free methods.
    Uses a repulsion-based approach: each point is pushed away from its neighbors.
    """
    if max_time is None:
        max_time = 10.0
    
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.1
    n_iter = 0
    
    while time.time() - start < max_time:
        # Compute the gradient of the covering radius
        # The covering radius is determined by the "worst" point on the sphere
        # which is the circumcenter of some facet.
        
        # Instead of computing the exact gradient, use a heuristic:
        # For each point, compute the sum of repulsion forces from nearby points.
        
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # The covering radius is roughly related to the minimum distance between
        # points (for uniform distributions).
        
        # Use a force-based approach: each point is repelled by nearby points
        # and attracted toward the center of the sphere (which is just normalization).
        
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < current_obj:  # Only consider points within the covering radius
                    # Repulsive force
                    f = (current_obj - d) / max(d, 1e-6)
                    forces[i] += f * (pts[i] - pts[j])
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update points
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size *= 0.9
        
        n_iter += 1
        if n_iter % 500 == 0:
            break
    
    return best_pts, best_obj

def _simulated_annealing(pts, max_time):
    """
    Simulated annealing to find a good configuration.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    temperature = 0.1
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < max_time:
        # Propose a random perturbation
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
            step_size = 0.01
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with a good initial configuration (Fibonacci spiral)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    # Local optimization
    deadline = _START_TIME + 140.0
    remaining = deadline - time.time()
    
    if remaining > 5:
        pts, obj = _local_optimize(pts, max_time=min(remaining - 5, 30))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Simulated annealing with multiple restarts
    remaining = deadline - time.time()
    if remaining > 10:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Create a perturbed version of the best solution
            perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * 0.5)
            pts, obj = _simulated_annealing(perturbed, max_time=min(remaining - 10, 20))
            
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
            
            # Also try local optimization from the SA result
            remaining = deadline - time.time()
            if remaining > 10:
                pts, obj = _local_optimize(pts, max_time=min(remaining - 10, 15))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
    
    # Final local optimization
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize(best_pts, max_time=min(remaining - 5, 20))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points and hemisphere issues
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        # Points are too close, add small perturbation
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v
    # such that v·x_i >= 0 for all i
    # This is equivalent to the origin being in the convex hull of the points
    # (by Gordan's theorem)
    
    # Check if the origin is in the convex hull
    # Use a linear programming approach or just check the convex hull
    try:
        hull = ConvexHull(best_pts)
        # Check if the origin is inside the convex hull
        # The origin is inside if all facets have the origin on the same side
        for simplex in hull.simplices:
            a = best_pts[simplex[0]]
            b = best_pts[simplex[1]]
            c = best_pts[simplex[2]]
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            # Check if origin is on the negative side of the facet
            if np.dot(normal, -a) > 0:
                # Origin is outside the convex hull, so points are in a hemisphere
                # Add a point at the antipode to fix this
                pass
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
