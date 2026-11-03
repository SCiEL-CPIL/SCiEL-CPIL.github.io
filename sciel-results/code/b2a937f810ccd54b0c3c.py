"""Improved covering radius minimization for 50 points on the unit sphere."""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

N = 50
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in radians using the convex hull approach.
    The covering radius is the maximum over all facets of the circumradius.
    For a facet (triangle) with vertices on the unit sphere, the circumradius
    in angular distance is the angle from the facet center (normalized) to a vertex.
    
    Actually, for a spherical cap covering problem:
    - Compute convex hull of the points
    - For each triangular facet, find the circumcenter of the triangle on the sphere
    - The angular distance from that circumcenter to any vertex of the triangle is the cap radius needed
    - The covering radius is the maximum such value
    
    More precisely: for each facet with vertices v1, v2, v3 on the unit sphere,
    the worst point in that facet's cap is the point equidistant (angularly) to all three vertices.
    This is the circumcenter of the spherical triangle.
    
    For a triangle on the unit sphere, the circumcenter (in the sense of equal angular distance)
    can be found as follows: the point c that maximizes min of angular distances to v1,v2,v3.
    This is equivalent to finding the point on the sphere equidistant from v1, v2, v3.
    
    A simpler approach: the covering radius is the max over facets of the circumradius.
    For a planar triangle with circumradius R (in 3D), if the triangle is inscribed in the unit sphere,
    the angular circumradius is arcsin(R) where R is the 3D circumradius... no.
    
    Let me think more carefully. Given three points v1, v2, v3 on the unit sphere forming a facet,
    the point on the sphere farthest from all three (within that facet's region) is the spherical circumcenter.
    
    The spherical circumcenter c satisfies |c-v1| = |c-v2| = |c-v3| in 3D Euclidean distance.
    Since c is on the unit sphere: |c|^2 = 1, |vi|^2 = 1.
    |c-vi|^2 = 2 - 2 c·vi.
    So c·v1 = c·v2 = c·v3 = const.
    
    This means c lies on the line through the origin and the circumcenter of the planar triangle v1,v2,v3.
    Let O be the circumcenter of the planar triangle (in 3D space). Then c = O / |O| (if O != 0).
    
    The angular distance from c to vi is arccos(c·vi).
    Since c·vi = (O·vi)/|O|, and O·vi = |O|^2 - (1/2)*... 
    
    Actually, for a triangle inscribed in a sphere of radius 1, the circumcenter of the planar triangle
    is at distance d from origin where d = sqrt(1 - R^2) where R is the circumradius of the planar triangle.
    Wait, that's not right either.
    
    Let me use a direct approach:
    - Find the circumcenter O of the planar triangle v1, v2, v3
    - The spherical circumcenter is c = O/|O|
    - The angular circumradius is arccos(c·v1) = arccos(O·v1/|O|)
    
    For a triangle with vertices on the unit sphere, if the planar circumcenter is O, then:
    |O - vi|^2 = R_planar^2 (same for all i)
    |vi|^2 = 1
    So |O|^2 - 2 O·vi + 1 = R_planar^2
    O·vi = (|O|^2 + 1 - R_planar^2) / 2
    
    And the angular distance from c=O/|O| to vi is:
    arccos(O·vi / |O|) = arccos((|O|^2 + 1 - R_planar^2) / (2|O|))
    
    This simplifies to: arccos((|O| + 1/|O| - R_planar^2/|O|) / 2)
    
    Actually, there's a cleaner formula. The circumradius of the spherical triangle (angular) is:
    arcsin(R_planar / |O|) ... let me verify.
    
    Hmm, let me just compute it numerically. For each facet:
    1. Compute planar circumcenter O
    2. c = O / |O|
    3. angular_radius = arccos(np.clip(c @ v1, -1, 1))
    
    The covering radius is the max over all facets.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: full hemisphere
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute planar circumcenter of triangle v1, v2, v3
        a = v2 - v1
        b = v3 - v1
        a_len2 = np.dot(a, a)
        b_len2 = np.dot(b, b)
        
        # Circumcenter formula
        # O = v1 + a * (b_len2 * |a|^2 - a·b * |b|^2) / (2 * (|a|^2 * |b|^2 - (a·b)^2)) ... 
        # Actually, standard 2D circumcenter in 3D:
        ab = np.dot(a, b)
        det = a_len2 * b_len2 - ab * ab
        if abs(det) < 1e-15:
            # Degenerate triangle
            continue
        
        # Circumcenter relative to v1:
        # t = (a_len2 * b_len2 - ab^2) ... no
        # Using: O = v1 + a * alpha + b * beta
        # where alpha, beta satisfy:
        # a·(a*alpha + b*beta) = |a|^2/2
        # b·(a*alpha + b*beta) = |b|^2/2
        # a_len2 * alpha + ab * beta = a_len2 / 2
        # ab * alpha + b_len2 * beta = b_len2 / 2
        
        alpha = (a_len2 * b_len2 - ab * ab) / (2 * det)  # wait this isn't right
        
        # Let me solve the 2x2 system:
        # [a_len2, ab] [alpha]   [a_len2/2]
        # [ab,     b_len2] [beta] = [b_len2/2]
        
        inv_det = 1.0 / det
        alpha = (a_len2 * 0.5 * b_len2 - ab * 0.5 * b_len2) * inv_det
        beta = (ab * 0.5 * a_len2 - b_len2 * 0.5 * a_len2) * inv_det
        
        # Wait, let me redo:
        # alpha = (a_len2/2 * b_len2 - ab * b_len2/2) / det
        # beta  = (ab * a_len2/2 - b_len2/2 * a_len2) / det
        
        # Hmm, Cramer's rule:
        # alpha = det([a_len2/2, ab; b_len2/2, b_len2]) / det
        #       = (a_len2/2 * b_len2 - ab * b_len2/2) / det
        #       = b_len2 * (a_len2 - ab) / (2 * det)
        
        # beta = det([a_len2, a_len2/2; ab, b_len2/2]) / det
        #      = (a_len2 * b_len2/2 - a_len2/2 * ab) / det
        #      = a_len2 * (b_len2 - ab) / (2 * det)
        
        alpha = b_len2 * (a_len2 - ab) / (2.0 * det)
        beta = a_len2 * (b_len2 - ab) / (2.0 * det)
        
        O = v1 + alpha * a + beta * b
        O_norm = np.linalg.norm(O)
        
        if O_norm < 1e-15:
            # Degenerate: triangle contains origin or is very small
            # The circumcenter is at infinity or origin; use max edge half-angle
            # For now, compute max angular distance to vertices from centroid
            centroid = (v1 + v2 + v3) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            for v in [v1, v2, v3]:
                ang = np.arccos(np.clip(np.dot(centroid, v), -1.0, 1.0))
                max_radius = max(max_radius, ang)
            continue
        
        c = O / O_norm
        # Angular distance from c to v1 (should be same for v2, v3)
        cos_ang = np.clip(np.dot(c, v1), -1.0, 1.0)
        ang = np.arccos(cos_ang)
        max_radius = max(max_radius, ang)
    
    return max_radius


def _smooth_objective(pts, beta):
    """Smooth approximation of max angular distance using log-sum-exp."""
    # For efficiency, compute pairwise angular distances
    # This is expensive for N=50, but we'll use it in gradient steps
    # Actually, for the objective, we just need the covering radius
    # For gradient-based optimization, we need a differentiable proxy
    
    # Compute all pairwise dot products
    dots = pts @ pts.T  # (N, N)
    # Angular distance: arccos(dot)
    # We want max over y on sphere of min_i arccos(y·xi)
    # This equals max over facets of circumradius
    
    # For a smooth proxy, we can use the log-sum-exp of the angular distances to nearest point
    # But this requires evaluating on a grid, which is expensive.
    
    # Alternative: use the convex hull circumradius as the objective (non-smooth but exact)
    # For gradient steps, we'll use a different approach.
    pass


def _gradient_step(pts, step_size, beta=500):
    """
    Perform a gradient descent step to reduce the covering radius.
    Uses the subgradient of the max circumradius.
    
    The idea: find the facet with the largest circumradius.
    The circumcenter of that facet is the "worst point" on the sphere.
    To reduce the covering radius, we should move the points of that facet
    away from the circumcenter (i.e., increase their angular distance to it).
    
    For a point xi on the unit sphere, the angular distance to a fixed point c is arccos(xi·c).
    To increase this, we want to decrease xi·c, i.e., move xi in the direction -c projected onto the tangent plane.
    
    But we need to be careful: moving points can affect other facets.
    A simple approach: for the worst facet, move its vertices slightly away from the circumcenter.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
    
    # Find the facet with maximum circumradius
    max_radius = 0.0
    worst_facet = None
    worst_c = None
    
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        a = v2 - v1
        b = v3 - v1
        a_len2 = np.dot(a, a)
        b_len2 = np.dot(b, b)
        ab = np.dot(a, b)
        det = a_len2 * b_len2 - ab * ab
        if abs(det) < 1e-15:
            continue
        
        alpha = b_len2 * (a_len2 - ab) / (2.0 * det)
        beta_c = a_len2 * (b_len2 - ab) / (2.0 * det)
        
        O = v1 + alpha * a + beta_c * b
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-15:
            continue
        
        c = O / O_norm
        cos_ang = np.clip(np.dot(c, v1), -1.0, 1.0)
        ang = np.arccos(cos_ang)
        
        if ang > max_radius:
            max_radius = ang
            worst_facet = simplex
            worst_c = c
    
    if worst_facet is None:
        return pts
    
    # Move the vertices of the worst facet away from the circumcenter
    # Direction: -c projected onto tangent plane at each vertex
    new_pts = pts.copy()
    for idx in worst_facet:
        xi = new_pts[idx]
        # Tangent direction: -c + (xi·c)xi (projection of -c onto tangent plane)
        xi_c = np.dot(xi, worst_c)
        tangent_dir = -worst_c + xi_c * xi
        tangent_norm = np.linalg.norm(tangent_dir)
        if tangent_norm > 1e-15:
            tangent_dir /= tangent_norm
            new_pts[idx] = xi + step_size * tangent_dir
    
    # Renormalize
    return _normalize(new_pts)


def _jittered_fibonacci_init(seed=42):
    """Generate jittered Fibonacci spiral points."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small jitter to break symmetry
    jitter = rng.randn(N, 3) * 0.01
    pts = pts + jitter
    return _normalize(pts)


def _local_search(pts, max_iters=2000, initial_step=0.01, decay=0.999):
    """
    Local search using gradient-like steps on the worst facet.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    step_size = initial_step
    
    for it in range(max_iters):
        new_pts = _gradient_step(best_pts, step_size, beta=500)
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-12:
            best_pts = new_pts
            best_radius = new_radius
        else:
            # Reduce step size
            step_size *= decay
            if step_size < 1e-8:
                break
    
    return best_pts, best_radius


def _random_restart_optimize(seed, time_budget):
    """
    Run a full optimization from a random restart.
    Returns (pts, radius).
    """
    t0 = time.time()
    
    # Initialize with jittered Fibonacci
    pts = _jittered_fibonacci_init(seed)
    
    # Phase 1: Coarse local search
    pts, radius = _local_search(pts, max_iters=1000, initial_step=0.02, decay=0.998)
    
    # Phase 2: Fine local search
    if time.time() - t0 < time_budget * 0.5:
        pts, radius = _local_search(pts, max_iters=2000, initial_step=0.005, decay=0.9995)
    
    # Phase 3: Very fine polish
    if time.time() - t0 < time_budget * 0.8:
        pts, radius = _local_search(pts, max_iters=5000, initial_step=0.001, decay=0.9998)
    
    return pts, radius


def construct():
    global N
    t_start = time.time()
    
    best_pts = None
    best_radius = np.pi
    
    # Try multiple random restarts
    n_restarts = 10
    for seed in range(n_restarts):
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT - 5:
            break
        
        remaining = TIME_LIMIT - 5 - elapsed
        per_restart = remaining / (n_restarts - seed)
        
        pts, radius = _random_restart_optimize(seed, per_restart)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
    
    # Final polish on the best result
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 10 and best_pts is not None:
        best_pts, best_radius = _local_search(best_pts, max_iters=10000, initial_step=0.0005, decay=0.9999)
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _jittered_fibonacci_init(0)
    
    # Final normalization and validation
    best_pts = _normalize(best_pts)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                # Separate them
                best_pts[j] = best_pts[i] + np.random.randn(3) * 0.01
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
