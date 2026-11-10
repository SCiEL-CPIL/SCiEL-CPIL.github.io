"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 135.0  # safety margin before 150s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _normalize(pts):
    """Normalize points to unit sphere, handling 1D/2D arrays."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        n = np.linalg.norm(pts)
        if n < 1e-12:
            return pts
        return pts / n
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-12] = 1.0
    return pts / norms[:, None]


def _covering_radius_fast(pts, n_samples=200000):
    """
    Estimate covering radius by sampling the sphere.
    Returns the max angular distance from any sample point to nearest config point.
    Uses vectorized computation for speed.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Generate sample points on sphere using stratified sampling
    # Use Fibonacci sphere for uniform coverage
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each sample to all config points
    # Use dot products: cos(theta) = dot(sample, point)
    # theta = arccos(dot), we want max theta
    
    # To avoid memory issues, process in batches
    batch_size = 50000
    max_cos = -2.0  # minimum possible cos value
    max_theta = 0.0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = samples[start:end]
        # Compute dot products: (batch_size, n)
        dots = batch @ pts.T
        # Clip for numerical safety
        dots = np.clip(dots, -1.0, 1.0)
        # Max cos per sample = min angular distance
        batch_max_cos = np.max(dots, axis=1)
        # Update global
        if np.min(batch_max_cos) > max_cos:
            max_cos = np.min(batch_max_cos)
    
    # Convert to angle
    max_cos = np.clip(max_cos, -1.0, 1.0)
    covering_radius_rad = np.arccos(max_cos)
    return np.degrees(covering_radius_rad)


def _covering_radius_hull(pts):
    """
    Compute covering radius using convex hull facets.
    For each facet, the circumcenter of the triangle (in 3D on sphere) gives
    the farthest point from the facet's vertices. The covering radius is the
    max over all facets of the angular distance from facet circumcenter to
    nearest config point.
    """
    from scipy.spatial import ConvexHull, Delaunay
    import numpy as np
    
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    
    facets = hull.simplices  # (m, 3)
    
    # For each triangular facet on the sphere, compute the "pole" - the point
    # on the sphere farthest from the facet plane (the normal direction)
    # The covering radius for that facet is the angular distance from the pole
    # to the nearest of the three vertices... actually it's the angular distance
    # from the pole to the nearest config point.
    
    # Actually, for a spherical triangle, the farthest point from the three vertices
    # is the spherical circumcenter. Let's compute it properly.
    
    # For a triangle with vertices a, b, c on unit sphere, the spherical circumcenter
    # is the intersection of the perpendicular bisectors.
    # A simpler approach: the point on the sphere that maximizes min distance to a,b,c
    # is the "spherical Fermat point" or circumcenter.
    
    # Alternative: for each facet, compute the outward normal, then the antipode of
    # the closest point on the facet... 
    
    # Actually, the standard result: the covering radius of a set of points on S^2
    # equals the maximum, over all facets of the convex hull, of the angular radius
    # of the circumcircle of that facet (where the circumcircle is the great circle
    # through the three vertices, and we take the arc that doesn't contain other points).
    
    # For a spherical triangle with vertices a, b, c (unit vectors), the spherical
    # circumcenter can be found as follows:
    # The circumcenter is the normalized cross product of (b-a) x (c-a) ... no.
    
    # Let me use a different approach: for each facet, find the point p on the sphere
    # that maximizes the minimum angular distance to the three vertices.
    # This is equivalent to finding the center of the largest empty circle inscribed
    # in the spherical triangle.
    
    # Simpler numerical approach: for each facet, sample the spherical triangle and
    # find the point with maximum minimum distance to the three vertices.
    
    # Even simpler: the covering radius is the max over all points y on sphere of
    # min_i angular_dist(y, x_i). The maximum occurs at a vertex of the Voronoi diagram
    # on the sphere, which corresponds to the circumcenters of the Delaunay triangles
    # (spherical Delaunay).
    
    # Let's just use the fast sampling method but with more samples near the hull facets.
    # Actually, let's just use the sampling method with adaptive refinement.
    
    return None  # Fall back to sampling


def _local_optimize(pts, max_iter=2000, tol=1e-10):
    """
    Local optimization using gradient descent on the sphere.
    Objective: minimize the covering radius.
    Approximate by maximizing the minimum distance to nearest neighbor,
    or by minimizing the sum of squared distances to nearest neighbors.
    """
    pts = np.asarray(pts, dtype=np.float64).copy()
    n = len(pts)
    
    # We'll use a repulsion-based approach: each point repels others,
    # with stronger repulsion for closer points.
    # This is similar to the Thomson problem but with a covering objective.
    
    # Use a soft-min approach: for each point, compute the distance to its
    # nearest neighbor, and we want to maximize the minimum of these.
    # Use log-sum-exp for differentiability.
    
    best_pts = pts.copy()
    best_score = _covering_radius_fast(pts, n_samples=50000)
    
    lr = 0.05
    momentum = 0.9
    vel = np.zeros_like(pts)
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        if it % 100 == 0 and _remaining() < 30.0:
            break
        
        # Compute pairwise distances
        # For efficiency, compute distances to all other points
        dots = pts @ pts.T
        np.fill_diagonal(dots, 1.0)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point, find its nearest neighbor distance
        # Set diagonal to inf
        dists_copy = dists.copy()
        np.fill_diagonal(dists_copy, np.inf)
        min_dists = np.min(dists_copy, axis=1)
        
        # Compute gradient: for each point i, the gradient of min_dists[i]
        # w.r.t. pts[i] points in the direction away from the nearest neighbor
        # For a smoother objective, use a soft version
        
        # Compute repulsion forces: each pair (i,j) with small distance repels strongly
        # Force on i from j: direction away from j, magnitude ~ 1/d^2 for small d
        forces = np.zeros_like(pts)
        for i in range(n):
            # Find nearest neighbors
            j_nearest = np.argmin(dists_copy[i])
            d = dists[i, j_nearest]
            if d < 1e-10:
                # Random perturbation
                forces[i] += np.random.randn(3) * 0.01
                continue
            # Direction from i to j
            dir_ij = pts[j_nearest] - pts[i]
            dir_ij_norm = np.linalg.norm(dir_ij)
            if dir_ij_norm < 1e-10:
                forces[i] += np.random.randn(3) * 0.01
                continue
            dir_ij = dir_ij / dir_ij_norm
            # Repel away from nearest neighbor
            # Magnitude: stronger when closer
            mag = 1.0 / (d + 0.1)
            forces[i] -= mag * dir_ij  # push away from j
        
        # Project forces onto tangent space
        for i in range(n):
            # Tangent space: perpendicular to pts[i]
            f_tangent = forces[i] - np.dot(forces[i], pts[i]) * pts[i]
            vel[i] = momentum * vel[i] + lr * f_tangent
            # Update
            pts[i] = pts[i] + vel[i]
            # Renormalize
            norm_i = np.linalg.norm(pts[i])
            if norm_i < 1e-10:
                pts[i] = np.random.randn(3)
                pts[i] /= np.linalg.norm(pts[i])
            else:
                pts[i] /= norm_i
        
        # Check for improvement
        if it % 50 == 0:
            score = _covering_radius_fast(pts, n_samples=50000)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        
        if _remaining() < 10.0:
            break
    
    # Final evaluation
    final_score = _covering_radius_fast(best_pts, n_samples=100000)
    return best_pts, final_score


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n):
    """Generate n random points on unit sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    pts /= norms[:, None]
    return pts


def construct():
    """
    Construct 50 points on the unit sphere to minimize covering radius.
    Returns array of shape (50, 3) with unit vectors.
    """
    t0 = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_score = _covering_radius_fast(pts, n_samples=100000)
    
    # Local optimization from multiple random starts
    n_starts = 10
    for start in range(n_starts):
        if _remaining() < 15.0:
            break
        
        if start == 0:
            # First start: Fibonacci
            current = _fibonacci_sphere(N)
        elif start == 1:
            # Second start: random
            current = _random_sphere(N)
        else:
            # Perturbed Fibonacci
            current = _fibonacci_sphere(N)
            current += np.random.randn(N, 3) * 0.1
            current = _normalize(current)
        
        # Local optimization
        opt_pts, opt_score = _local_optimize(current, max_iter=1500)
        
        if opt_score < best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
        
        if _remaining() < 15.0:
            break
    
    # Final refinement with more iterations if time allows
    if _remaining() > 10.0:
        final_pts, final_score = _local_optimize(best_pts, max_iter=1000)
        if final_score < best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Ensure output is valid
    best_pts = _normalize(best_pts)
    
    # Verify no coincident points and not in a hemisphere
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 1.0)
    if np.max(np.abs(dots - 1.0)) > 1e-6:
        # Points are too close, add small perturbation
        best_pts += np.random.randn(N, 3) * 1e-4
        best_pts = _normalize(best_pts)
    
    # Check not in a closed hemisphere: the origin should be inside the convex hull
    # Quick check: sum of points should be near zero for good coverage
    centroid = np.mean(best_pts, axis=0)
    if np.linalg.norm(centroid) > 0.3:
        # Might be in a hemisphere, try to fix
        best_pts = _fibonacci_sphere(N)
        best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
