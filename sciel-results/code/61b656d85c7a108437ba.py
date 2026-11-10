"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def _normalize(pts):
    """Normalize rows of pts to unit length, handling near-zero norms."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _angular_dist(a, b):
    """Angular distance in radians between unit vectors a and b (scalar)."""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees for a set of unit-sphere points.
    The covering radius is the max angular distance from any sphere point to
    its nearest site. This equals the max circumradius of the convex hull facets
    (for points not in a hemisphere). We compute it via the convex hull.
    """
    pts = _normalize(pts)
    # Check if all points are in a closed hemisphere
    # A set of points on the sphere is in a closed hemisphere iff the origin is
    # not in the interior of their convex hull.
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Degenerate: maybe all points coincide or too few
        # Fallback: compute max nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        # Sample many points on the sphere
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(20000, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.degrees(np.max(dists))

    # The covering radius is determined by the largest empty cap, which
    # corresponds to the facet with the largest circumradius (angular).
    # For each facet, the covering point is the center of the largest empty
    # spherical cap, which is the circumcenter of the facet (pushed out to the
    # sphere). The angular distance from the facet's circumcenter to any vertex
    # of the facet is the covering radius for that facet.
    
    # Get facet vertices
    facets = hull.simplices
    max_circ_rad = 0.0
    
    for face in facets:
        # Get the points defining this facet
        face_pts = pts[face]
        # For a spherical cap defined by k points, the center of the largest
        # empty cap is the point on the sphere equidistant (in angular sense)
        # to all k points, on the side away from the origin.
        # For 3 points (triangle facet), this is the circumcenter of the triangle
        # on the sphere.
        
        # The circumcenter of points on the sphere: find unit vector c such that
        # c·p_i = constant for all i in face, and |c|=1.
        # This means c is in the direction of the sum of p_i (for 3 points forming
        # an acute triangle on the sphere).
        
        # More robustly: solve for c such that c·(p_i - p_j) = 0 for all pairs,
        # i.e., c is orthogonal to the plane of the facet (in 3D Euclidean space).
        # The circumcenter on the sphere is along the normal to the facet plane.
        
        if len(face) == 3:
            p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
            # Normal to the plane
            n = np.cross(p1 - p0, p2 - p0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # The circumcenter on the sphere is ±n. We want the one that is
            # farthest from the origin side, i.e., the one where the cap is empty.
            # Check which direction gives a point that is "outside" the hull.
            # The outward normal of the convex hull facet points away from the hull interior.
            # For a convex hull of points on the unit sphere, the outward normal
            # points away from the origin.
            
            # Determine sign: the center should be on the side where the cap is empty.
            # The midpoint of the facet is inside the hull. The circumcenter on the
            # sphere should be on the opposite side of the facet from the hull interior.
            mid = np.mean(face_pts, axis=0)
            # The outward direction is the one where dot(n, mid) < 0 (since mid is
            # inside the hull, and the facet normal pointing outward has negative
            # dot with the inward direction).
            # Actually, for a convex hull of points on the unit sphere, the origin
            # is inside the hull (if not in a hemisphere). The outward normal points
            # away from the origin, so dot(n_outward, origin) = 0, but the direction
            # is away from the centroid of the hull.
            
            # Simpler: the circumcenter c = ±n. The angular distance from c to p_i
            # is arccos(c·p_i). We want the c that maximizes the minimum distance
            # to the sites... no, we want the covering radius which is the max over
            # all sphere points of min distance to sites. For a facet, the worst point
            # is the circumcenter of the facet (the point equidistant to all facet
            # vertices and farthest from all other sites).
            
            # The angular distance from the circumcenter to each vertex is the same.
            # Let's compute it for both signs and pick the one that gives a valid
            # circumcenter (i.e., the point on the sphere in the direction of the
            # outward normal).
            
            # The outward normal of a convex hull facet: the facet is part of the
            # boundary, and the hull is on one side. For points on the unit sphere,
            # the hull contains the origin (if not in a hemisphere), so the outward
            # normal points away from the origin. Since the origin is at 0, the
            # direction is simply the normal pointing away from the hull's interior.
            
            # A robust way: the center of the largest empty cap for this facet is
            # the point on the sphere in the direction of the outward normal. The
            # angular distance from this center to any vertex of the facet is the
            # circumradius for this facet.
            
            # To determine the outward normal: the hull's interior is on the side
            # where the origin is. So the outward normal n_out satisfies
            # n_out · (p - origin) > 0 for p outside, and the origin is on the
            # "inside" side. Since origin = 0, we need another point inside the hull.
            # The centroid of all points is inside the hull.
            centroid = np.mean(pts, axis=0)
            # The outward normal points away from the centroid (roughly).
            if np.dot(n, centroid) > 0:
                n = -n
            
            # Now n is the outward normal. The circumcenter is c = n (unit vector).
            c = n
            # Angular distance from c to each vertex
            dots = np.clip(face_pts @ c, -1.0, 1.0)
            circ_rad = np.arccos(np.min(dots))  # min dot = max distance
            max_circ_rad = max(max_circ_rad, circ_rad)
        else:
            # For non-triangular facets (shouldn't happen in 3D convex hull with
            # general position, but handle just in case)
            # Use the same approach with the normal
            p0 = face_pts[0]
            n = np.cross(face_pts[1] - p0, face_pts[2] - p0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            centroid = np.mean(pts, axis=0)
            if np.dot(n, centroid) > 0:
                n = -n
            c = n
            dots = np.clip(face_pts @ c, -1.0, 1.0)
            circ_rad = np.arccos(np.min(dots))
            max_circ_rad = max(max_circ_rad, circ_rad)
    
    return np.degrees(max_circ_rad)


def _covering_radius_fast(pts):
    """
    Faster covering radius estimate using sampling + local refinement.
    """
    pts = _normalize(pts)
    from scipy.spatial import cKDTree
    
    tree = cKDTree(pts)
    
    # Sample points on the sphere
    rng = np.random.default_rng(12345)
    n_samples = 50000
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    dists, _ = tree.query(samples)
    worst_idx = np.argmax(dists)
    worst_point = samples[worst_idx]
    worst_dist = dists[worst_idx]
    
    # Refine: find the actual worst point by looking at the circumcenter of the
    # nearest neighbors of the worst sample point
    # Get the 3 nearest sites to the worst point
    _, nn_idx = tree.query(worst_point, k=3)
    nn_pts = pts[nn_idx]
    
    # The worst point should be the circumcenter of these 3 points (if they form
    # a facet) or in the interior of the spherical triangle
    p0, p1, p2 = nn_pts[0], nn_pts[1], nn_pts[2]
    n = np.cross(p1 - p0, p2 - p0)
    n_norm = np.linalg.norm(n)
    if n_norm > 1e-12:
        n = n / n_norm
        # Determine outward direction
        centroid = np.mean(pts, axis=0)
        if np.dot(n, centroid) > 0:
            n = -n
        c = n
        # Check if c is farther from sites than worst_point
        dist_c = np.arccos(np.clip(c @ pts, -1.0, 1.0))
        min_dist_c = np.min(dist_c)
        if min_dist_c > worst_dist:
            return np.degrees(min_dist_c)
    
    return np.degrees(worst_dist)


def _log_sum_exp_covering(pts, temperature=0.05):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimization.
    """
    pts = _normalize(pts)
    # Sample points on the sphere
    rng = np.random.default_rng(999)
    n_samples = 20000
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute angular distances from each sample to all sites
    # dist[i, j] = arccos(samples[i] · pts[j])
    dots = np.clip(samples @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_samples, N)
    
    # For each sample, the distance to the nearest site
    min_dists = np.min(dists, axis=1)  # (n_samples,)
    
    # Log-sum-exp approximation of the max
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    lse = temperature * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return lse


def _gradient_covering(pts, temperature=0.05):
    """
    Compute gradient of the log-sum-exp covering radius approximation.
    """
    pts = _normalize(pts)
    rng = np.random.default_rng(999)
    n_samples = 20000
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    dots = np.clip(samples @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # Gradient with respect to each point
    # d/dp_j of min_i arccos(s·p_j) for the nearest site
    # This is complex; use numerical gradient as fallback
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with a good initial configuration
    # Use a combination of icosahedron-based and Fibonacci spiral
    def _initial_config():
        # Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    
    best_pts = _initial_config()
    best_score = _covering_radius_fast(best_pts)
    
    # Try to improve using gradient-based optimization on the log-sum-exp objective
    # We'll use a simple projected gradient method
    
    def _objective_and_grad(x):
        """
        Objective: log-sum-exp covering radius approximation.
        x is a flat array of length 3*N.
        """
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        
        # Use a fixed set of sample points for consistent gradient
        rng = np.random.default_rng(42)
        n_samples = 10000
        samples = rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        dots = np.clip(samples @ pts.T, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        temp = 0.05
        scaled = min_dists / temp
        max_scaled = np.max(scaled)
        weights = np.exp(scaled - max_scaled)
        weights /= np.sum(weights)
        
        obj = temp * (max_scaled + np.log(np.sum(weights)))
        
        # Gradient: d(obj)/d(pts[j])
        # obj = temp * log(sum_i exp(min_k arccos(s_i · p_k) / temp))
        # This is complex to differentiate exactly. Use numerical gradient.
        return obj
    
    # Use scipy minimize with L-BFGS-B on the log-sum-exp objective
    # But we need a proper gradient. Let's use a numerical approach.
    
    # Alternative: use the fact that the covering radius is determined by the
    # convex hull facets. We can do local optimization by moving points to
    # reduce the largest facet circumradius.
    
    # Strategy: iterative local optimization
    # 1. Compute the covering radius and identify the worst facet(s)
    # 2. Move the points involved in the worst facet to reduce its circumradius
    # 3. Repeat
    
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        pts = _normalize(best_pts)
        
        # Compute covering radius via convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        facets = hull.simplices
        max_circ_rad = 0.0
        worst_facet = None
        
        for face in facets:
            face_pts = pts[face]
            if len(face) != 3:
                continue
            p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
            n = np.cross(p1 - p0, p2 - p0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            centroid = np.mean(pts, axis=0)
            if np.dot(n, centroid) > 0:
                n = -n
            c = n
            dots = np.clip(face_pts @ c, -1.0, 1.0)
            circ_rad = np.arccos(np.min(dots))
            if circ_rad > max_circ_rad:
                max_circ_rad = circ_rad
                worst_facet = face
        
        if worst_facet is None:
            break
        
        # Move the points in the worst facet slightly toward the circumcenter
        # to reduce the circumradius
        face_pts = pts[list(worst_facet)]
        p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        centroid = np.mean(pts, axis=0)
        if np.dot(n, centroid) > 0:
            n = -n
        
        # Move each point in the facet toward the circumcenter by a small step
        step_size = 0.01
        new_pts = pts.copy()
        for idx in worst_facet:
            direction = n - pts[idx]
            direction = direction / np.linalg.norm(direction)
            new_pts[idx] = pts[idx] + step_size * direction
        
        new_pts = _normalize(new_pts)
        
        # Check if this improved the covering radius
        new_score = _covering_radius_fast(new_pts)
        if new_score < best_score:
            best_pts = new_pts
            best_score = new_score
    
    # Final: try a few random perturbations and keep the best
    rng = np.random.default_rng(777)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        perturbed = best_pts + rng.normal(scale=0.05, size=(N, 3))
        perturbed = _normalize(perturbed)
        score = _covering_radius_fast(perturbed)
        if score < best_score:
            best_pts = perturbed
            best_score = score
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
