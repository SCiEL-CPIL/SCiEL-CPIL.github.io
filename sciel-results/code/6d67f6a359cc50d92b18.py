"""Rewritten program: optimize 50-point spherical covering using gradient-based relaxation."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# Deadline guard: kill after 150s wall-clock, keep 10s safety margin
_DEADLINE_SECONDS = 140.0


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of unit vectors on the sphere.
    Uses the convex hull: the covering radius is the maximum over all facets of the
    circumradius of the spherical triangle corresponding to that facet.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        raise ValueError(f"Expected shape ({N}, 3), got {pts.shape}")
    
    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        raise ValueError("Non-finite values in points")
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                raise ValueError("Coincident points detected")
    
    # Check if all points are in a closed hemisphere (invalid)
    # A valid configuration must have the origin in the interior of the convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Degenerate hull - likely in a hemisphere
        raise ValueError("Degenerate convex hull - points may be in a closed hemisphere")
    
    # Check if origin is inside the convex hull
    # The origin should be strictly inside for a valid covering
    # We check by seeing if all facet normals point away from origin
    normals = hull.equations[:, :-1]  # normal vectors
    # For a valid configuration, the origin should be on the negative side of all facets
    # i.e., dot(normal, origin) < offset for all facets
    # hull.equations[i] = [n, b] means n.x + b = 0, with interior being n.x + b <= 0
    # So for origin: 0 + b <= 0, i.e., b <= 0
    # If any b > 0, origin is outside that facet's half-space
    offsets = hull.equations[:, -1]
    if np.any(offsets > 1e-10):
        raise ValueError("Origin outside convex hull - configuration in a closed hemisphere")
    
    # Compute covering radius from hull facets
    # Each facet is a triangle (or polygon) on the sphere
    # The circumradius of the spherical cap corresponding to a facet is the angular
    # distance from the facet center (in the spherical sense) to the vertices
    # 
    # For a planar triangle with vertices a, b, c on the unit sphere, the circumcenter
    # of the spherical triangle is in the direction of the cross product or via the
    # formula for the circumradius of a spherical triangle.
    # 
    # A simpler approach: for each facet, compute the circumcenter of the planar triangle
    # formed by the three vertices, then project to the sphere. The angular distance from
    # this center to any vertex is the circumradius of that facet.
    # 
    # Actually, the covering radius is the maximum over all points on the sphere of the
    # distance to the nearest site. The worst point is at the circumcenter of some facet
    # of the Voronoi diagram, which corresponds to a facet of the convex hull.
    # 
    # For a facet with vertices v1, v2, v3 (unit vectors), the circumcenter in the
    # spherical sense is the point c on the unit sphere such that the angular distances
    # to v1, v2, v3 are equal. This is the direction of the vector that is equidistant
    # (in angle) to all three.
    # 
    # A practical approach: the circumcenter of the spherical triangle is proportional to
    # the sum of unit vectors pointing from the centroid toward each vertex, but more
    # accurately, we can solve for the point c on the unit sphere minimizing max angle.
    # 
    # Simpler: for each facet, compute the planar circumcenter of the triangle formed by
    # the three vertices, normalize to get a direction, and compute the max angular distance
    # from that direction to the three vertices.
    
    covering_radius = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            # Skip non-triangular facets (shouldn't happen for convex hull in 3D)
            continue
        v1, v2, v3 = pts[simplex]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the direction of the vector that is equidistant to v1, v2, v3
        # This is the normalized version of the solution to:
        #   c . v1 = c . v2 = c . v3
        #   |c| = 1
        # 
        # One approach: the circumcenter is proportional to:
        #   v1 * (v2 x v3) + v2 * (v3 x v1) + v3 * (v1 x v2)
        # This is the direction of the circumcenter of the planar triangle projected.
        
        cross_23 = np.cross(v2, v3)
        cross_31 = np.cross(v3, v1)
        cross_12 = np.cross(v1, v2)
        
        circum_vec = v1 * np.linalg.norm(cross_23) + v2 * np.linalg.norm(cross_31) + v3 * np.linalg.norm(cross_12)
        
        norm_circum = np.linalg.norm(circum_vec)
        if norm_circum < 1e-15:
            # Degenerate facet
            continue
        circum_center = circum_vec / norm_circum
        
        # Compute angular distances from circum_center to each vertex
        for v in [v1, v2, v3]:
            dot = np.clip(np.dot(circum_center, v), -1.0, 1.0)
            angle = np.arccos(dot)
            covering_radius = max(covering_radius, angle)
    
    return covering_radius


def _soft_covering_radius(pts, beta=100.0):
    """
    Smooth surrogate for the covering radius using a log-sum-exp / soft-max approach.
    This provides a differentiable approximation to the max angular distance.
    """
    pts = np.asarray(pts, dtype=np.float64)
    
    # Sample points on the sphere to approximate the max
    # Use a set of test points
    n_test = 5000
    # Generate test points using Fibonacci spiral
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each test point to each site
    # dist[i, j] = angular distance from test_pts[i] to pts[j]
    dots = np.clip(test_pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each test point, find the min distance to any site
    min_dists = np.min(dists, axis=1)
    
    # Soft max: beta * log(sum(exp(beta * min_dists)))
    # For numerical stability, subtract max
    max_d = np.max(min_dists)
    log_sum = np.log(np.sum(np.exp(beta * (min_dists - max_d))))
    soft_max = max_d + np.log(np.sum(np.exp(beta * (min_dists - max_d)))) / beta
    
    return soft_max


def _gradient_step(pts, beta=50.0, lr=0.01, n_test=2000):
    """
    Perform a gradient descent step to reduce the soft covering radius.
    Uses automatic differentiation via finite differences or analytic gradients.
    """
    pts = np.asarray(pts, dtype=np.float64)
    
    # Generate test points
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances
    dots = np.clip(test_pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each test point, find nearest site
    nearest_idx = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(n_test), nearest_idx]
    
    # Soft max gradient
    max_d = np.max(min_dists)
    weights = np.exp(beta * (min_dists - max_d))
    weights /= np.sum(weights)
    
    # Gradient of soft max w.r.t. each point
    # d/d(pts[j]) soft_max = sum_i weights[i] * d(min_dist_i)/d(pts[j])
    # d(min_dist_i)/d(pts[j]) is nonzero only if j == nearest_idx[i]
    # d(arccos(x))/dx = -1/sqrt(1-x^2)
    # x = test_pts[i] . pts[j]
    # d(x)/d(pts[j]) = test_pts[i]
    
    grad = np.zeros_like(pts)
    for i in range(n_test):
        j = nearest_idx[i]
        if weights[i] < 1e-15:
            continue
        dot_ij = np.dot(test_pts[i], pts[j])
        dot_ij = np.clip(dot_ij, -1.0 + 1e-12, 1.0 - 1e-12)
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        denom = np.sqrt(1.0 - dot_ij * dot_ij)
        if denom < 1e-12:
            continue
        grad[j] -= weights[i] * beta * test_pts[i] / denom
    
    # Project gradient to tangent space (remove radial component)
    for j in range(N):
        radial = np.dot(grad[j], pts[j])
        grad[j] -= radial * pts[j]
    
    # Gradient descent step
    new_pts = pts - lr * grad
    
    # Renormalize to unit sphere
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / norms
    
    return new_pts


def _lloyd_iteration(pts, n_iters=500):
    """
    Lloyd's algorithm for spherical k-means / covering.
    """
    pts = np.asarray(pts, dtype=np.float64)
    
    for _ in range(n_iters):
        # Compute Voronoi cells (approximate via nearest neighbor)
        # For each point, compute the centroid of its Voronoi cell
        # Approximate by sampling
        n_sample = 20000
        i = np.arange(n_sample) + 0.5
        z = 1.0 - 2.0 * i / n_sample
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Assign each sample to nearest site
        dots = samples @ pts.T
        nearest = np.argmax(dots, axis=1)
        
        # Compute new centers as mean of assigned samples (on sphere)
        new_pts = np.zeros_like(pts)
        for j in range(N):
            mask = nearest == j
            if np.sum(mask) > 0:
                new_pts[j] = np.sum(samples[mask], axis=0)
                norm = np.linalg.norm(new_pts[j])
                if norm > 1e-15:
                    new_pts[j] /= norm
                else:
                    new_pts[j] = pts[j]
            else:
                new_pts[j] = pts[j]
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < 1e-10:
            break
    
    return pts


def _refine_with_gradient(pts, max_iters=2000, beta_start=20.0, beta_end=200.0):
    """
    Refine the configuration using gradient descent on the soft covering radius.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n_total = max_iters
    
    for it in range(n_total):
        # Anneal beta
        t = it / max(n_total - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Adaptive learning rate
        lr = 0.05 * (1.0 - 0.5 * t)
        
        new_pts = _gradient_step(pts, beta=beta, lr=lr, n_test=3000)
        
        # Check improvement
        old_cr = _soft_covering_radius(pts, beta=beta)
        new_cr = _soft_covering_radius(new_pts, beta=beta)
        
        if new_cr < old_cr:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = pts - 0.1 * (new_pts - pts)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            pts = new_pts
    
    return pts


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Lloyd iterations
    remaining = _DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 5:
        n_lloyd = min(100, int(remaining / 0.5))
        pts = _lloyd_iteration(pts, n_iters=n_lloyd)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 2: Gradient refinement
    remaining = _DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 5:
        n_grad = min(1500, int(remaining / 0.05))
        pts = _refine_with_gradient(pts, max_iters=n_grad)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 3: Final Lloyd polish
    remaining = _DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 3:
        n_lloyd2 = min(200, int(remaining / 0.2))
        pts = _lloyd_iteration(best_pts, n_iters=n_lloyd2)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Final check
    remaining = _DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 2:
        # One more gradient pass
        pts = _refine_with_gradient(best_pts, max_iters=min(500, int(remaining / 0.05)))
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
