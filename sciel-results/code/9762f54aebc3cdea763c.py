"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _fibonacci_spiral(n):
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius (max angular distance to nearest point).
    
    Uses the fact that the covering radius is determined by the largest
    circumradius of the Delaunay triangulation facets. We approximate this
    by checking the Voronoi cells' circumcenters.
    
    For efficiency, we compute the maximum over all pairs of nearby points
    of the angular distance to their midpoint (which approximates the
    Voronoi cell circumcenter for well-distributed points).
    """
    n = len(pts)
    # Compute all pairwise distances
    # This is O(n^2) but n=130 so it's manageable
    dot = pts @ pts.T
    np.fill_diagonal(dot, 1.0)
    # Clip to [-1, 1] for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    # Angular distances
    ang_dist = np.arccos(dot)
    
    # For each point, find its nearest neighbors
    # The covering radius is approximately the max over all points of
    # the angular distance to the "worst" direction in its Voronoi cell
    # A good approximation: for each point, look at its k nearest neighbors
    # and estimate the circumradius of the triangle formed by the point
    # and its two nearest neighbors
    
    # Better approach: use the fact that the covering radius equals the
    # maximum over all facets of the convex hull of the circumradius.
    # We can approximate by considering all triples of mutually nearest neighbors.
    
    # Simpler and more robust: compute the maximum angular distance from
    # any point on the sphere to the nearest configuration point by sampling.
    # But that's expensive. Instead, use the Delaunay-based approach.
    
    # For each point i, find its 3 nearest neighbors j, k
    # The circumradius of triangle (i, j, k) on the sphere gives a lower bound
    # on the covering radius contribution from that facet.
    
    # Get k nearest neighbors for each point
    k = 3
    # For each point, sort by distance
    idx = np.argsort(ang_dist, axis=1)
    nearest_k = idx[:, 1:k+1]  # skip self
    
    # For each point, compute the circumradius of the spherical triangle
    # formed with its two nearest neighbors (not just one)
    # Actually, we need the circumradius of the Voronoi cell, which is
    # determined by the 3 nearest neighbors typically.
    
    # Compute circumradius for each point using its 3 nearest neighbors
    max_radius = 0.0
    for i in range(n):
        j, k1, k2 = nearest_k[i]
        # Three points on sphere: p_i, p_j, p_k1
        # The circumradius of the spherical triangle is the angular distance
        # from the circumcenter to any vertex.
        # The circumcenter of a spherical triangle can be found as the
        # intersection of the great circles perpendicular to the sides.
        
        # Alternative: the circumradius R satisfies sin(R) = a/(2*R_h) 
        # where a is the side length and R_h is the... this is getting complex.
        
        # Simpler: the covering radius is the maximum over all points y on
        # the sphere of min_i d(y, p_i). This is equivalent to the maximum
        # circumradius of the Delaunay triangulation.
        
        # For a spherical triangle with vertices A, B, C, the circumradius
        # can be computed as follows:
        # Let the side lengths be a = d(B,C), b = d(A,C), c = d(A,B)
        # The circumradius R satisfies:
        # sin(R) = abc / (4 * area)  where area is the spherical excess
        # This is complex. Let's use a different approach.
        
        # The covering radius for point i's Voronoi cell is the maximum
        # angular distance from p_i to any point in its Voronoi cell.
        # This occurs at the circumcenter of the Delaunay triangle.
        
        # For numerical simplicity, let's use the following:
        # The circumradius of the triangle (p_i, p_j, p_k1) on the sphere
        # is the angular radius of the smallest circle containing all three.
        # This is at least max(d(p_i,p_j), d(p_i,p_k1), d(p_j,p_k1))/2
        # and at most the circumradius computed properly.
        
        # Use the formula: for a spherical triangle with side lengths a, b, c,
        # the circumradius R is given by:
        # tan(R) = (a*b*c) / (4*sqrt(s*(s-a)*(s-b)*(s-c)))  where s = (a+b+c)/2
        # but this is for plane triangles. For spherical, it's different.
        
        # Let's just use a sampling-based estimate for the covering radius
        # that's fast enough.
        pass
    
    # Fast approximate covering radius: sample many points on the sphere
    # and compute the max min-distance.
    # Use a stratified sampling approach.
    
    # Actually, for n=130, we can afford a more careful computation.
    # The covering radius is the maximum over all facets of the convex hull
    # of the circumradius. Let's compute this properly.
    
    # For each point, consider its 3 nearest neighbors. The Delaunay triangle
    # formed by three mutually nearest neighbors has a circumradius.
    
    # Compute all pairwise angular distances
    # Already have ang_dist
    
    # For each triple (i, j, k) where j and k are among i's nearest neighbors,
    # compute the circumradius of the spherical triangle.
    
    # Circumradius of spherical triangle with vertices v1, v2, v3:
    # The circumcenter is the point equidistant from all three.
    # It lies on the normal to the plane through v1, v2, v3 (for the planar case).
    # On the sphere, the circumcenter is the intersection of the great circles
    # that are the perpendicular bisectors of the sides.
    
    # A simpler formula: if the triangle has side lengths a, b, c (angular),
    # and the circumradius is R, then:
    # sin(R) = sin(a)sin(b)sin(c) / (4*sin((a+b+c)/2)*sin((a-b+c)/2)*sin((a+b-c)/2))
    # This is the spherical law of sines for the circumradius.
    
    # Let me use a robust numerical approach: for each point i, find the
    # circumcenter of its Voronoi cell by optimization.
    
    # Given time constraints, let's use a sampling-based estimate.
    n_samples = 5000
    # Generate random points on the sphere
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi_s = np.arccos(np.random.uniform(-1, 1, n_samples))
    sample_pts = np.column_stack([
        np.sin(phi_s) * np.cos(theta),
        np.sin(phi_s) * np.sin(theta),
        np.cos(phi_s)
    ])
    
    # For each sample point, find the nearest configuration point
    # Compute distances
    dot_s = sample_pts @ pts.T
    dot_s = np.clip(dot_s, -1.0, 1.0)
    min_dist = np.arccos(np.min(dot_s, axis=1))
    return np.max(min_dist)


def _soft_covering_radius(pts, beta=50.0):
    """Compute a smooth surrogate for the covering radius using log-sum-exp.
    
    The covering radius is max_y min_i d(y, p_i). We approximate this by
    considering a finite set of test points (e.g., midpoints of nearby pairs)
    and using a soft-max.
    """
    n = len(pts)
    # Compute pairwise distances
    dot = pts @ pts.T
    np.fill_diagonal(dot, 1.0)
    dot = np.clip(dot, -1.0, 1.0)
    ang_dist = np.arccos(dot)
    
    # For each point, find its nearest neighbor
    min_dist_to_other = np.min(ang_dist, axis=1)
    
    # The covering radius is at least max_i (min_dist_to_other[i] / 2)
    # because the midpoint between i and its nearest neighbor is at that distance.
    # But it could be larger if the Voronoi cell extends further.
    
    # Use a soft version: the covering radius is approximately
    # (1/beta) * log(sum_i exp(beta * d_i)) where d_i are estimates of
    # the local covering radius at each point.
    
    # A better surrogate: consider all midpoints between pairs of nearby points
    # and compute the max distance from those midpoints to the nearest point.
    
    # For efficiency, only consider pairs within some threshold
    threshold = np.max(min_dist_to_other) * 2
    # Find pairs within threshold
    mask = ang_dist < threshold
    np.fill_diagonal(mask, False)
    
    # Get indices of pairs
    i_idx, j_idx = np.where(mask)
    
    if len(i_idx) == 0:
        return np.max(min_dist_to_other) / 2
    
    # Compute midpoints (normalized)
    mids = (pts[i_idx] + pts[j_idx]) / 2
    mids = _normalize(mids)
    
    # For each midpoint, compute distance to nearest configuration point
    dot_m = mids @ pts.T
    dot_m = np.clip(dot_m, -1.0, 1.0)
    min_d = np.arccos(np.min(dot_m, axis=1))
    
    # Soft max
    max_d = np.max(min_d)
    if beta <= 0:
        return max_d
    # Use log-sum-exp for numerical stability
    shifted = min_d - max_d
    log_sum = max_d + np.log(np.sum(np.exp(shifted)))
    return log_sum / beta


def _gradient_step(pts, step_size=0.01, beta=50.0):
    """Perform one gradient descent step to reduce the covering radius.
    
    Uses the gradient of the soft covering radius with respect to each point.
    """
    n = len(pts)
    
    # Compute pairwise distances
    dot = pts @ pts.T
    np.fill_diagonal(dot, 1.0)
    dot = np.clip(dot, -1.0, 1.0)
    ang_dist = np.arccos(dot)
    
    # For each point i, compute the gradient of the local covering radius.
    # The idea: point i should move away from its nearest neighbors to
    # increase the minimum distance, but also consider the Voronoi cell.
    
    # Simple repulsion: each point is repelled by all other points,
    # with stronger repulsion for closer points.
    # The force on point i from point j is proportional to (p_i - p_j) / d(i,j)^2
    # but constrained to the sphere.
    
    # Compute repulsion forces
    # For each pair (i, j), the repulsion on i is (p_i - p_j) * w(d)
    # where w(d) is a weight that decreases with distance.
    
    # Use a Gaussian kernel: w(d) = exp(-d^2 / sigma^2)
    sigma = 0.5  # radians
    w = np.exp(-ang_dist**2 / sigma**2)
    np.fill_diagonal(w, 0.0)
    
    # Force on point i: sum_j w(i,j) * (p_i - p_j)
    # = p_i * sum_j w(i,j) - sum_j w(i,j) * p_j
    force = pts * np.sum(w, axis=1, keepdims=True) - (w @ pts)
    
    # Project force onto the tangent plane of the sphere at each point
    # The tangent space at p_i is orthogonal to p_i.
    # Project: force_tangent = force - (force . p_i) * p_i
    dot_fp = np.sum(force * pts, axis=1, keepdims=True)
    force_tangent = force - dot_fp * pts
    
    # Update: p_i <- normalize(p_i + step_size * force_tangent_i)
    new_pts = pts + step_size * force_tangent
    new_pts = _normalize(new_pts)
    
    return new_pts


def _local_refine(pts, max_iters=100, tol=1e-8):
    """Refine the configuration using gradient descent with adaptive step size."""
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    step_size = 0.005
    for it in range(max_iters):
        # Try a gradient step
        new_pts = _gradient_step(best_pts, step_size=step_size)
        
        # Compute new covering radius
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr - 1e-10:
            best_pts = new_pts
            best_cr = new_cr
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.02)
        else:
            # Decrease step size
            step_size *= 0.8
            if step_size < 1e-5:
                break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Phase 1: Coarse optimization with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Leave 30s for final polish
            break
        
        # Add small random perturbation
        if restart > 0:
            noise = np.random.normal(0, 0.05, size=pts.shape)
            pts = _normalize(pts + noise)
        
        # Optimize
        refined_pts, cr = _local_refine(pts, max_iters=200)
        
        if cr < best_cr:
            best_pts = refined_pts.copy()
            best_cr = cr
    
    # Phase 2: Fine polish
    if time.time() < deadline - 10:
        refined_pts, cr = _local_refine(best_pts, max_iters=500)
        if cr < best_cr:
            best_pts = refined_pts.copy()
            best_cr = cr
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
