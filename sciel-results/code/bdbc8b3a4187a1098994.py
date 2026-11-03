"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math


def _normalize(pts):
    """Normalize points to unit sphere, handling near-zero norms."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees for a set of points on the unit sphere.
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest point in pts.
    
    This equals the maximum circumradius of the facets of the convex hull,
    where each facet is a triangle (or we can use the Delaunay triangulation).
    
    For a set of points on the sphere, the covering radius is determined by the
    largest empty spherical cap. The boundaries of these caps are arcs of great circles.
    The worst point is either at the center of a facet of the convex hull (for
    triangular facets) or at a vertex.
    
    Actually, the covering radius for points on the sphere is the maximum over
    all facets of the convex hull of the circumradius of that facet. For a
    triangular facet with vertices a, b, c on the unit sphere, the circumcenter
    (in 3D) is the point equidistant from a, b, c. The circumradius is the angle
    from this circumcenter to any of the vertices.
    
    However, we need to be careful: the circumcenter of a spherical triangle
    (on the unit sphere) is the normalized version of the 3D circumcenter of the
    three points, IF the 3D circumcenter is on the same side as the facet normal.
    
    Let me think more carefully. The covering radius is the maximum distance from
    any point on the sphere to the nearest data point. The Voronoi cells on the
    sphere are bounded by great-circle arcs. The worst point in each Voronoi cell
    is either a vertex (intersection of three arcs) or a point on an edge.
    
    A simpler approach: use the convex hull. The facets of the convex hull of the
    points correspond to the "empty" regions. For each facet (triangle) with
    vertices a, b, c, the circumcenter of the spherical triangle is the point on
    the sphere equidistant from a, b, c. The angular distance from this circumcenter
    to a (or b, or c) gives a candidate covering radius.
    
    To find the circumcenter of three points a, b, c on the unit sphere:
    - The circumcenter is in the plane containing a, b, c, on the sphere.
    - It's the normalized version of the 3D circumcenter of the triangle abc,
      provided the 3D circumcenter is on the correct side.
    
    Actually, let me use a more robust approach. For each facet of the convex hull,
    compute the circumradius of the spherical triangle.
    
    For a spherical triangle with vertices a, b, c (unit vectors), the circumcenter
    can be found as follows:
    - The circumcenter lies in the plane of a, b, c.
    - Let n = (b-a) x (c-a) be the normal to the plane.
    - The circumcenter (in the plane) satisfies |x - a| = |x - b| = |x - c|.
    - In 3D, the circumcenter of triangle abc is:
      Let d = 2 * (a x b) . (b x c) + 2 * (b x c) . (c x a) + 2 * (c x a) . (a x b)
      Actually, let me use a standard formula.
      
    For three points a, b, c in 3D, the circumcenter is:
    Let v1 = b - a, v2 = c - a
    Let d = 2 * (v1 x v2)
    Let t = (|a|^2 * (v1 x v2) + |b|^2 * (v2 x v0) + |c|^2 * (v0 x v1)) / |d|^2
    where v0 = c - b... this is getting complicated.
    
    Let me use a simpler numerical approach. For each facet, I can find the point
    on the sphere that maximizes the minimum distance to the three vertices.
    
    Actually, the cleanest approach: for each triangular facet of the convex hull,
    compute the circumcenter of the three vertices in 3D, normalize it to get a
    point on the sphere, and check that it's "outside" the convex hull (i.e., on
    the correct side). The angular distance from this circumcenter to any vertex
    is the circumradius of the spherical triangle.
    
    Let me implement this step by step.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a grid-based approach
        return _covering_radius_grid(pts)
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of triangle abc in 3D
        # Using the formula: circumcenter = (|a|^2*(b-c) + |b|^2*(c-a) + |c|^2*(a-b)) / (2*(a-b) x (a-c))
        # Since |a|=|b|=|c|=1, this simplifies.
        
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        cross_norm_sq = np.dot(cross, cross)
        
        if cross_norm_sq < 1e-14:
            # Degenerate triangle
            continue
        
        # Circumcenter in the plane of the triangle
        # For unit vectors, the circumcenter is:
        # c0 = (a * (b x c) + b * (c x a) + c * (a x b)) / |a x b + b x c + c x a| ... no
        
        # Let me use the standard formula for circumcenter of three points:
        # Let d = 2 * ((b-a) x (c-a))
        # c0 = (|a|^2 * ((b-a) x (c-a)) + |b|^2 * ((c-a) x (a-b)) + |c|^2 * ((a-b) x (c-a))) / |d|^2
        # Since |a|=|b|=|c|=1:
        # c0 = (1 * ((b-a) x (c-a)) + 1 * ((c-a) x (a-b)) + 1 * ((a-b) x (c-a))) / |d|^2
        
        ba = a - b
        cb = b - c
        
        num = np.cross(b - a, c - a) + np.cross(c - a, a - b) + np.cross(a - b, c - a)
        # Wait, let me be more careful.
        # Standard formula: circumcenter = a + t, where t is in the plane spanned by ab, ac.
        # t = alpha * ab + beta * ac
        # |a + t - b|^2 = |t|^2  =>  |t - ab|^2 = |t|^2  =>  -2 t . ab + |ab|^2 = 0  =>  t . ab = |ab|^2 / 2
        # |a + t - c|^2 = |t|^2  =>  |t - ac|^2 = |t|^2  =>  -2 t . ac + |ac|^2 = 0  =>  t . ac = |ac|^2 / 2
        
        # Solve: [ab . ab, ab . ac; ac . ab, ac . ac] [alpha; beta] = [|ab|^2/2; |ac|^2/2]
        A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                      [np.dot(ac, ab), np.dot(ac, ac)]])
        rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
        
        det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
        if abs(det) < 1e-14:
            continue
        
        alpha = (rhs[0] * A[1,1] - rhs[1] * A[0,1]) / det
        beta = (A[0,0] * rhs[1] - A[1,0] * rhs[0]) / det
        
        t = alpha * ab + beta * ac
        circumcenter_3d = a + t
        circumcenter_norm = np.linalg.norm(circumcenter_3d)
        
        if circumcenter_norm < 1e-12:
            continue
        
        circumcenter_sphere = circumcenter_3d / circumcenter_norm
        
        # Check that the circumcenter is on the "outside" of the convex hull
        # The facet normal points outward. The circumcenter should be on the same side.
        # The facet normal is cross(ab, ac) normalized, pointing outward.
        facet_normal = cross / np.linalg.norm(cross)
        
        # Check: the circumcenter should be such that it's "beyond" the facet from the origin.
        # Actually, for the covering radius, we want the circumcenter to be the point
        # that is farthest from all points, which should be on the outside.
        
        # Compute the angular distance from circumcenter_sphere to a
        cos_angle = np.dot(circumcenter_sphere, a)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        if angle > max_circumradius:
            max_circumradius = angle
    
    # Convert to degrees
    return math.degrees(max_circumradius)


def _covering_radius_grid(pts):
    """Fallback covering radius computation using a grid."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Create a grid of points on the sphere
    n_theta = 200
    n_phi = 400
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute minimum distance from each grid point to any data point
    # Use batch processing to avoid memory issues
    max_min_angle = 0.0
    batch_size = 10000
    for start in range(0, len(grid_pts), batch_size):
        end = min(start + batch_size, len(grid_pts))
        batch = grid_pts[start:end]
        # Compute dot products
        dots = batch @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # Maximum dot product = minimum angle
        angles = np.arccos(min_dots)
        max_min_angle = max(max_min_angle, np.max(angles))
    
    return math.degrees(max_min_angle)


def _initial_config():
    """Generate initial Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _refine_lloyd(pts, time_budget=100.0):
    """
    Refine the configuration using a Lloyd-like algorithm.
    At each step, find the worst facet (largest circumradius) and move the
    vertices toward the circumcenter of that facet.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Precompute some parameters
    step_size = 0.1
    
    for iteration in range(10000):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_circumradius = 0.0
        worst_simplex = None
        worst_circumcenter = None
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            cross_norm_sq = np.dot(cross, cross)
            
            if cross_norm_sq < 1e-14:
                continue
            
            A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                          [np.dot(ac, ab), np.dot(ac, ac)]])
            rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
            
            det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
            if abs(det) < 1e-14:
                continue
            
            alpha = (rhs[0] * A[1,1] - rhs[1] * A[0,1]) / det
            beta = (A[0,0] * rhs[1] - A[1,0] * rhs[0]) / det
            
            t = alpha * ab + beta * ac
            circumcenter_3d = a + t
            circumcenter_norm = np.linalg.norm(circumcenter_3d)
            
            if circumcenter_norm < 1e-12:
                continue
            
            circumcenter_sphere = circumcenter_3d / circumcenter_norm
            
            cos_angle = np.dot(circumcenter_sphere, a)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            
            if angle > max_circumradius:
                max_circumradius = angle
                worst_simplex = simplex
                worst_circumcenter = circumcenter_sphere
        
        if worst_simplex is None:
            break
        
        # Move the vertices of the worst facet toward the circumcenter
        new_pts = pts.copy()
        for idx in worst_simplex:
            # Move a fraction of the way toward the circumcenter
            direction = worst_circumcenter - pts[idx]
            new_pts[idx] = pts[idx] + step_size * direction
        
        new_pts = _normalize(new_pts)
        
        # Check if the new configuration is better
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius:
            best_pts = new_pts.copy()
            best_radius = new_radius
            pts = new_pts
        else:
            # If not better, reduce step size
            step_size *= 0.9
            if step_size < 1e-4:
                break
    
    return best_pts


def _refine_repulsive(pts, time_budget=80.0):
    """
    Refine using repulsive forces: each point is repelled by nearby points,
    but attracted toward the center of the worst Voronoi cell.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    learning_rate = 0.05
    
    for iteration in range(5000):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Compute repulsive forces
        # For each point, compute the gradient of the covering radius approximation
        # Use a smooth approximation: sum of exp(-beta * min_distance^2) for each point
        
        # Actually, let's use a simpler approach: for each point, move it away from
        # its neighbors, and toward the circumcenter of the worst facet.
        
        # Compute all pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_circumradius = 0.0
        worst_simplex = None
        worst_circumcenter = None
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            cross_norm_sq = np.dot(cross, cross)
            
            if cross_norm_sq < 1e-14:
                continue
            
            A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                          [np.dot(ac, ab), np.dot(ac, ac)]])
            rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
            
            det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
            if abs(det) < 1e-14:
                continue
            
            alpha = (rhs[0] * A[1,1] - rhs[1] * A[0,1]) / det
            beta = (A[0,0] * rhs[1] - A[1,0] * rhs[0]) / det
            
            t = alpha * ab + beta * ac
            circumcenter_3d = a + t
            circumcenter_norm = np.linalg.norm(circumcenter_3d)
            
            if circumcenter_norm < 1e-12:
                continue
            
            circumcenter_sphere = circumcenter_3d / circumcenter_norm
            
            cos_angle = np.dot(circumcenter_sphere, a)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            
            if angle > max_circumradius:
                max_circumradius = angle
                worst_simplex = simplex
                worst_circumcenter = circumcenter_sphere
        
        if worst_simplex is None:
            break
        
        # Move points: repel from neighbors, attract worst facet vertices to circumcenter
        new_pts = pts.copy()
        
        # Repulsion: for each point, compute the sum of repulsive forces from neighbors
        # Use a cutoff: only repel from points within some angular distance
        for i in range(N):
            force = np.zeros(3)
            for j in range(N):
                if i == j:
                    continue
                d = angles[i, j]
                if d < 0.5:  # Within ~28 degrees
                    # Repulsive force proportional to 1/d^2
                    force -= pts[j] / max(d**2, 1e-6)
            
            # Attraction for worst facet vertices
            if i in worst_simplex:
                force += 2.0 * (worst_circumcenter - pts[i])
            
            new_pts[i] = pts[i] + learning_rate * force
        
        new_pts = _normalize(new_pts)
        
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius:
            best_pts = new_pts.copy()
            best_radius = new_radius
            pts = new_pts
        else:
            learning_rate *= 0.95
            if learning_rate < 1e-5:
                break
    
    return best_pts


def construct():
    """
    Construct a set of 130 points on the unit sphere that minimizes the covering radius.
    Returns an array of shape (130, 3) of unit vectors.
    """
    start_time = time.time()
    total_time_budget = 140.0  # Safety margin before 150s limit
    
    # Generate initial configuration
    pts = _initial_config()
    
    # Phase 1: Lloyd-like refinement targeting worst facets
    time_left = total_time_budget - (time.time() - start_time)
    if time_left > 10:
        pts = _refine_lloyd(pts, time_budget=min(time_left - 5, 60))
    
    # Phase 2: Repulsive force refinement
    time_left = total_time_budget - (time.time() - start_time)
    if time_left > 10:
        pts = _refine_repulsive(pts, time_budget=min(time_left - 5, 60))
    
    # Final check
    pts = _normalize(pts)
    
    # Ensure the result is valid
    if not np.all(np.isfinite(pts)):
        pts = _initial_config()
    
    return pts
# EVOLVE-BLOCK-END
