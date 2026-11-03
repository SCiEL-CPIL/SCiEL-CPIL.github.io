"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _compute_covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in pts. This equals the largest circumradius of the facets
    of the convex hull (in angular terms).
    
    We use the fact that the covering radius is the maximum over all spherical Voronoi
    vertices (circumcenters of Delaunay triangles on the sphere) of the distance to
    the nearest site. For a set of points on the sphere, the covering radius is
    max over all empty caps of the cap radius, which equals the max circumradius
    of the spherical Delaunay triangulation.
    
    A simpler approach: the covering radius is the maximum over all pairs of points
    of half the angular distance between them, if those two points are adjacent in
    the nearest-neighbor graph... Actually, the correct way is:
    
    The covering radius = max over all points y on sphere of min_i angle(y, p_i)
    
    This is equivalent to: for each facet (triangle) of the convex hull of the points,
    compute the circumcenter (in 3D, projected to sphere) and the distance from that
    circumcenter to each of the three vertices. The covering radius is the maximum
    of these distances over all facets.
    
    But computing the convex hull and checking all facets is complex. Instead, we can
    use the fact that the worst point is either:
    1. A circumcenter of a Delaunay triangle
    2. A point on an edge (midpoint of the longest edge in some sense)
    
    Actually, a robust numerical approach: the covering radius is the maximum of
    the distances from the circumcenters of all spherical Delaunay triangles to their
    vertices. But we can approximate this by considering all triples and finding
    the empty circumcircles.
    
    For efficiency and correctness, I'll use a different approach:
    The covering radius is max_i (1/2 * max_j angle(p_i, p_j)) is NOT correct.
    
    Let me use the proper method: The covering radius equals the maximum, over all
    points x on the sphere, of min_i ||x - p_i|| (in angular terms). The maximum
    occurs at a point that is equidistant to at least 3 points (a Voronoi vertex)
    or equidistant to 2 points (on an edge of the Voronoi diagram).
    
    For a practical computation: compute the convex hull of the 50 points in 3D.
    For each triangular face, compute the circumcenter of that triangle (the point
    equidistant to all 3 vertices in 3D space, projected to the sphere). Check if
    this circumcenter is "empty" (no other point is closer to it than to the 3 face
    vertices). The covering radius is the max distance from such a circumcenter to
    its nearest point.
    
    Actually, the simplest correct approach for the covering radius:
    It's the maximum over all spherical Voronoi vertices of the distance to the
    nearest site. Spherical Voronoi vertices correspond to circumcenters of Delaunay
    triangles. 
    
    For a robust implementation, I'll compute it as follows:
    1. For each triple of points, compute the circumcenter on the sphere.
    2. Check if it's a valid Voronoi vertex (the cap centered there with radius =
       distance to the 3 points contains no other points).
    3. The covering radius is the max such distance.
    
    But this is O(n^3) which for n=50 is 19600 triples, each requiring O(n) check,
    so ~1 million operations. That's fine.
    
    Actually, let me use a simpler but still correct method:
    The covering radius is the largest value r such that there exists a point y on
    the sphere with min_i angle(y, p_i) >= r. This is equivalent to finding the
    largest empty cap.
    
    For numerical purposes, I'll compute: for each pair of points, find the midpoint
    on the great circle, and for each triple, find the circumcenter. The covering
    radius is the max over all these candidates of the distance to the nearest point.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute pairwise angular distances
    # angle = arccos(dot)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)  # in radians
    
    # The covering radius is at least half the maximum nearest-neighbor distance
    # and at most half the maximum pairwise distance.
    
    # Find all candidate Voronoi vertices:
    # 1. Midpoints of edges (pairs where the geodesic midpoint is a local max of min-distance)
    # 2. Circumcenters of triples
    
    best_r = 0.0
    
    # For each triple, compute circumcenter on sphere
    # The circumcenter of three points on the unit sphere is the normalized
    # cross product of (p2-p1) and (p3-p1), or more robustly, the point equidistant
    # to all three.
    
    # A point x on the unit sphere equidistant to p_i, p_j, p_k satisfies:
    # x . p_i = x . p_j = x . p_k
    # This means x is perpendicular to (p_i - p_j) and (p_i - p_k)
    # So x is parallel to (p_i - p_j) x (p_i - p_k)
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = np.cross(v1, v2)
                norm = np.linalg.norm(cross)
                if norm < 1e-10:
                    continue
                # Two possible circumcenters
                c1 = cross / norm
                c2 = -c1
                
                for c in (c1, c2):
                    # Check if c is a valid Voronoi vertex: the cap centered at c
                    # with radius = angle(c, pts[i]) should contain no other points
                    # closer than pts[i], pts[j], pts[k]
                    dist_to_i = np.arccos(np.clip(np.dot(c, pts[i]), -1.0, 1.0))
                    # Check all other points
                    dots_c = np.clip(c @ pts, -1.0, 1.0)
                    all_dists = np.arccos(dots_c)
                    # The three points i, j, k should be the nearest
                    min_dist = np.min(all_dists)
                    # c is a Voronoi vertex if min_dist == dist_to_i (approximately)
                    # and the three nearest are i, j, k
                    if abs(min_dist - dist_to_i) < 1e-8:
                        # Count how many points are at this minimum distance
                        count_near = np.sum(np.abs(all_dists - min_dist) < 1e-8)
                        if count_near >= 3:
                            best_r = max(best_r, min_dist)
    
    # Also check midpoints of edges as candidates
    # For each pair, the geodesic midpoint
    for i in range(n):
        for j in range(i + 1, n):
            # Midpoint on the shorter arc
            d = angles[i, j]
            if d > np.pi:
                d = 2 * np.pi - d  # take shorter arc
            # Midpoint direction
            # m = normalize(sin((pi-d)/2)/sin(d) * p_i + sin(d/2)/sin(d) * p_j)
            # Simpler: m = normalize(p_i + p_j) if d < pi
            if d < np.pi - 1e-10:
                m = pts[i] + pts[j]
                nm = np.linalg.norm(m)
                if nm < 1e-10:
                    continue
                m = m / nm
                dots_m = np.clip(m @ pts, -1.0, 1.0)
                all_dists = np.arccos(dots_m)
                min_dist = np.min(all_dists)
                # Check if i and j are the two nearest
                count_near = np.sum(np.abs(all_dists - min_dist) < 1e-8)
                if count_near >= 2:
                    best_r = max(best_r, min_dist)
    
    return np.degrees(best_r)


def _riesz_energy(pts, s=20.0):
    """Compute Riesz s-energy: sum over pairs of 1/d^s where d is angular distance."""
    pts = _normalize(pts)
    n = len(pts)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    # Avoid self-pairs
    np.fill_diagonal(dots, 2.0)
    dists = np.arccos(dots)
    # Energy = sum_{i<j} 1/d_{ij}^s
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid division by zero
    d = np.maximum(d, 1e-15)
    return np.sum(1.0 / (d ** s))


def _negative_riesz_energy(pts, s=20.0):
    return -_riesz_energy(pts, s)


def _gradient_riesz(pts, s=20.0):
    """Compute gradient of Riesz s-energy w.r.t. points on the sphere."""
    pts = _normalize(pts)
    n = len(pts)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(dots, 2.0)
    dists = np.arccos(dots)
    
    # d/dp_i [1/d_{ij}^s] where d = arccos(p_i . p_j)
    # d/dd [1/d^s] = -s / d^{s+1}
    # d/dp_i [arccos(p_i . p_j)] = -p_j / sqrt(1 - (p_i . p_j)^2)
    # So d/dp_i [1/d_{ij}^s] = -s / d^{s+1} * (-p_j / sqrt(1 - dots_{ij}^2))
    #                        = s * p_j / (d^{s+1} * sqrt(1 - dots_{ij}^2))
    
    # But we need the gradient on the tangent space. The gradient of the energy
    # w.r.t. p_i (in R^3) projected to the tangent plane at p_i is:
    # grad_i = sum_{j != i} s * p_j / (d_{ij}^{s+1} * sqrt(1 - dots_{ij}^2))
    # Then the Riemannian gradient is: grad_i - (grad_i . p_i) * p_i
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            dot = dots[i, j]
            sqrt_term = np.sqrt(max(1.0 - dot * dot, 1e-15))
            if d < 1e-15:
                continue
            coeff = s / (d ** (s + 1) * sqrt_term)
            grad[i] += coeff * pts[j]
        # Project to tangent space
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return grad


def _optimize_from_start(start_pts, s=20.0, max_iter=500, time_limit=100.0):
    """Optimize Riesz s-energy minimization from a starting configuration."""
    start_time = time.time()
    pts = _normalize(start_pts.copy())
    
    # Use L-BFGS-B with the tangent-space gradient
    # We parameterize by the points themselves, but need to re-normalize after each step.
    # Alternative: use a different parameterization.
    
    # For simplicity, use scipy's minimize with the constraint that points are on the sphere.
    # We'll use the approach of minimizing the energy with a penalty for leaving the sphere,
    # or better, use the fact that we can project back to the sphere after each gradient step.
    
    # Let's use a custom optimization loop with gradient descent + projection
    lr = 0.01
    best_energy = _riesz_energy(pts, s)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        g = _gradient_riesz(pts, s)
        
        # Gradient descent step
        pts_new = pts - lr * g
        pts_new = _normalize(pts_new)
        
        e_new = _riesz_energy(pts_new, s)
        
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
            pts = pts_new
            # Decrease learning rate slowly
            lr = max(lr * 0.999, 0.0001)
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_pts, best_energy


def _compute_covering_radius_fast(pts):
    """
    Fast computation of covering radius using the convex hull approach.
    For 50 points, we can compute the convex hull and check each face.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute all pairwise distances
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # The covering radius is the maximum over all spherical Voronoi vertices
    # of the distance to the nearest site.
    
    # Efficient approach: The covering radius equals the maximum circumradius
    # of the Delaunay triangulation on the sphere.
    
    # For a set of points on the sphere, the Delaunay triangulation can be
    # obtained from the convex hull. Each face of the convex hull corresponds
    # to a Delaunay triangle.
    
    # Let's use a simpler but effective approach:
    # The covering radius is bounded below by half the maximum nearest-neighbor
    # distance and we can compute it exactly by checking all candidate circumcenters.
    
    best_r = 0.0
    
    # Check all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = np.cross(v1, v2)
                norm = np.linalg.norm(cross)
                if norm < 1e-12:
                    continue
                c = cross / norm
                # Check if this is an empty circumcenter
                dots_c = np.clip(c @ pts, -1.0, 1.0)
                all_dists = np.arccos(dots_c)
                min_dist = np.min(all_dists)
                
                # The circumcenter is valid if the three points i,j,k are among
                # the nearest points to c
                idx_i, idx_j, idx_k = i, j, k
                if (all_dists[idx_i] <= min_dist + 1e-10 and
                    all_dists[idx_j] <= min_dist + 1e-10 and
                    all_dists[idx_k] <= min_dist + 1e-10):
                    best_r = max(best_r, min_dist)
    
    # Also check edge midpoints
    for i in range(n):
        for j in range(i + 1, n):
            d = angles[i, j]
            if d >= np.pi - 1e-10:
                continue
            m = pts[i] + pts[j]
            nm = np.linalg.norm(m)
            if nm < 1e-12:
                continue
            m = m / nm
            dots_m = np.clip(m @ pts, -1.0, 1.0)
            all_dists = np.arccos(dots_m)
            min_dist = np.min(all_dists)
            if (all_dists[i] <= min_dist + 1e-10 and
                all_dists[j] <= min_dist + 1e-10):
                best_r = max(best_r, min_dist)
    
    return np.degrees(best_r)


def _fibonacci_sphere(n):
    """Generate n points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses Riesz s-energy minimization with high s to approximate packing,
    then polishes with covering-radius-aware optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_pts = None
    best_cr = np.inf
    
    # Try multiple starting configurations
    starts = []
    starts.append(_fibonacci_sphere(N))
    for seed in [1, 2, 3, 4, 5]:
        starts.append(_random_sphere(N, seed))
    
    # Optimize each start using Riesz s-energy
    s = 20.0
    for start in starts:
        if time.time() > deadline:
            break
        pts, energy = _optimize_from_start(start, s=s, max_iter=300, time_limit=20.0)
        
        if time.time() > deadline:
            break
        
        cr = _compute_covering_radius_fast(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Polish the best result with a second stage of optimization
    if best_pts is not None and time.time() < deadline - 10.0:
        # Try a lower s for more uniform distribution
        for s2 in [10.0, 5.0]:
            if time.time() > deadline - 5.0:
                break
            pts2, _ = _optimize_from_start(best_pts, s=s2, max_iter=200, time_limit=15.0)
            cr2 = _compute_covering_radius_fast(pts2)
            if cr2 < best_cr:
                best_cr = cr2
                best_pts = pts2.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
