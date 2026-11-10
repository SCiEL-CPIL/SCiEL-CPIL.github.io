"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Known best covering radius in radians (Hardin, Sloane, Smith 1994)
KNOWN_BEST_RAD = np.radians(18.3000226)


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest point in pts.
    
    This equals the maximum circumradius of the facets of the convex hull,
    where each facet is a triangle (or higher-dim simplex) formed by 3+ points.
    For a set of points on the unit sphere, the covering radius is the maximum
    angular distance from any point on the sphere to its nearest data point.
    
    We compute this by finding, for each facet of the convex hull, the center
    of the circumscribed cap, and the angular radius of that cap.
    """
    # Ensure points are on unit sphere
    pts = _normalize(pts.copy())
    
    # Check for coincident points
    d = cdist(pts, pts)
    if np.any(d < 1e-10):
        # Coincident points - invalid
        return np.pi  # Return worst possible
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # If hull computation fails, fall back to a grid-based estimate
        return _covering_radius_grid(pts)
    
    # For each facet, compute the circumradius
    # A facet is defined by its simplices (triangles in 3D)
    # The covering radius is the maximum over all points on the sphere of the
    # distance to the nearest data point.
    
    # The farthest points from all data points are at the centers of the
    # "Voronoi cells" on the sphere. For a convex hull of points on the unit
    # sphere, the covering radius is determined by the facets.
    
    # For each facet (triangle in 3D), compute the circumcenter of the triangle
    # projected onto the sphere, and the angular radius.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # simplex is an array of indices into pts
        tri_pts = pts[simplex]
        
        # Compute the circumcenter of the triangle
        # For 3 points on the unit sphere, the circumcenter of the spherical triangle
        # is in the direction of the normalized sum of the points (for acute triangles)
        # More generally, we need to find the point on the sphere equidistant from all 3.
        
        # Method: find the point c on the unit sphere that maximizes the minimum
        # angular distance to the 3 triangle vertices. This is the center of the
        # largest empty circle in the spherical triangle.
        
        # For a triangle on the sphere, the circumcenter is along the direction
        # perpendicular to the plane of the triangle, but only if the triangle is
        # acute. For obtuse triangles, the center is at the midpoint of the longest edge.
        
        # Let's compute the plane normal
        a, b, c = tri_pts[0], tri_pts[1], tri_pts[2]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        # The circumcenter of the spherical triangle lies along the normal direction
        # (or its opposite) if the triangle is "acute" in the spherical sense.
        # The angular radius is the angle between the normal and any vertex.
        
        # Check which side of the plane the origin is on
        # The center should be on the same side as the origin for the cap to be valid
        
        # Compute angle between normal and each vertex
        cos_angles = np.abs(np.array([np.dot(a, normal), np.dot(b, normal), np.dot(c, normal)]))
        
        # The circumradius is the angle from the normal to the farthest vertex
        # But we need to be careful: the normal might point away from the cap center
        
        # Determine the correct direction: the cap center should be such that
        # the cap contains no other data points. The center is along +normal or -normal.
        
        # Let's try both directions
        for sign in [1, -1]:
            center_dir = sign * normal
            # Compute angular distances from center_dir to each vertex
            dists = np.arccos(np.clip(np.abs(np.array([np.dot(a, center_dir),
                                                         np.dot(b, center_dir),
                                                         np.dot(c, center_dir)])), -1, 1))
            # The radius of this cap is the max distance to the vertices
            cap_radius = np.max(dists)
            
            # Check if this cap is "empty" - i.e., no other data points are inside
            # A point p is inside the cap if the angle from center_dir to p < cap_radius
            # AND p is on the same side of the plane
            other_pts = np.delete(pts, simplex, axis=0)
            if len(other_pts) > 0:
                cos_to_center = np.clip(np.dot(other_pts, center_dir), -1, 1)
                angles_to_center = np.arccos(cos_to_center)
                # Points inside the cap have angle < cap_radius
                inside = np.sum(angles_to_center < cap_radius - 1e-10)
                if inside > 0:
                    # This cap is not empty, so it's not a valid covering facet
                    continue
            
            max_radius = max(max_radius, cap_radius)
    
    if max_radius < 1e-10:
        # Fallback: use grid-based estimate
        return _covering_radius_grid(pts)
    
    return max_radius


def _covering_radius_grid(pts, n_grid=200):
    """
    Estimate covering radius using a grid of points on the sphere.
    This is a fallback when convex hull method fails.
    """
    pts = _normalize(pts.copy())
    
    # Generate a grid of points on the sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi)
    grid_pts = np.column_stack([
        np.sin(THETA) * np.cos(PHI).ravel(),
        np.sin(THETA) * np.sin(PHI).ravel(),
        np.cos(THETA).ravel()
    ])
    
    # For each grid point, find the nearest data point
    d = cdist(grid_pts, pts)
    # Angular distance = arccos of dot product
    dot = np.sum(grid_pts * pts.T, axis=1)  # (n_grid_pts, N)
    dot = np.clip(dot, -1, 1)
    angular_dists = np.arccos(dot)
    min_angular_dists = np.min(angular_dists, axis=1)
    
    return np.max(min_angular_dists)


def _local_optimize(pts, max_iter=200, step_size=0.01, n_restarts=1):
    """
    Local optimization: gradient-free optimization to minimize covering radius.
    Uses a simple iterative approach: for each point, try moving it in small
    steps and keep the move if it improves the covering radius.
    """
    pts = _normalize(pts.copy())
    best_cr = _covering_radius_grid(pts, n_grid=100)  # Use grid for speed during optimization
    
    for restart in range(n_restarts):
        for iteration in range(max_iter):
            improved = False
            for i in range(N):
                # Try moving point i in random directions
                for _ in range(8):
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    new_pts = pts.copy()
                    new_pts[i] = pts[i] + step_size * direction
                    new_pts[i] = _normalize(new_pts[i:i+1])[0]
                    
                    # Check for coincident points
                    if np.min(cdist(new_pts, new_pts)) < 1e-6:
                        continue
                    
                    new_cr = _covering_radius_grid(new_pts, n_grid=80)
                    if new_cr < best_cr - 1e-6:
                        pts = new_pts
                        best_cr = new_cr
                        improved = True
                        break
            
            if not improved:
                # Try smaller step size
                step_size *= 0.5
                if step_size < 1e-5:
                    break
    
    return pts, _covering_radius(pts)


def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    return _normalize(pts)


def _antipodal_pair_optimize(pts, max_iter=50):
    """
    Optimize by considering pairs of points that are nearly antipodal.
    For each such pair, try to move them slightly to improve coverage.
    """
    pts = _normalize(pts.copy())
    best_cr = _covering_radius_grid(pts, n_grid=100)
    
    for iteration in range(max_iter):
        # Find pairs with large angular separation
        d = cdist(pts, pts)
        dot = np.clip(d, -1, 1)
        angular_dists = np.arccos(dot)
        
        # Find the pair with maximum angular distance
        i, j = np.unravel_index(np.argmax(angular_dists), angular_dists.shape)
        if i == j:
            break
        
        # Try moving both points slightly toward each other's antipode
        for di in [-0.01, 0, 0.01]:
            for dj in [-0.01, 0, 0.01]:
                new_pts = pts.copy()
                # Move point i slightly
                if di != 0:
                    dir_i = -pts[i] + di * np.random.randn(3)
                    dir_i /= np.linalg.norm(dir_i)
                    new_pts[i] = _normalize(new_pts[i:i+1] + 0.01 * dir_i)[0]
                # Move point j slightly
                if dj != 0:
                    dir_j = -pts[j] + dj * np.random.randn(3)
                    dir_j /= np.linalg.norm(dir_j)
                    new_pts[j] = _normalize(new_pts[j:j+1] + 0.01 * dir_j)[0]
                
                new_cr = _covering_radius_grid(new_pts, n_grid=80)
                if new_cr < best_cr - 1e-6:
                    pts = new_pts
                    best_cr = new_cr
    
    return pts, _covering_radius(pts)


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    
    # First, do local optimization from the Fibonacci start
    if time.time() < deadline:
        opt_pts, opt_cr = _local_optimize(pts, max_iter=100, step_size=0.02, n_restarts=1)
        if opt_cr < best_cr:
            best_pts = opt_pts
            best_cr = opt_cr
    
    # Try several random initializations
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Different random seeds
        seed = 1000 + trial
        rand_pts = _random_points(N, seed=seed)
        
        # Local optimization from random start
        opt_pts, opt_cr = _local_optimize(rand_pts, max_iter=50, step_size=0.03, n_restarts=1)
        
        if opt_cr < best_cr:
            best_pts = opt_pts
            best_cr = opt_cr
    
    # Final refinement with smaller steps
    if time.time() < deadline - 5.0:
        final_pts, final_cr = _local_optimize(best_pts, max_iter=100, step_size=0.005, n_restarts=1)
        if final_cr < best_cr:
            best_pts = final_pts
            best_cr = final_cr
    
    # Ensure output is valid
    best_pts = _normalize(best_pts)
    
    # Final check
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
