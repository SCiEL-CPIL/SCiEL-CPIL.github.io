"""Rewritten program: optimize 130-point spherical covering via repulsion + local search."""
import numpy as np
import time

N = 130


def _fibonacci_spiral(n, seed=0):
    """Generate n points on unit sphere via Fibonacci spiral."""
    if seed != 0:
        rng = np.random.default_rng(seed)
        offset = rng.random()
    else:
        offset = 0.5
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in degrees) of a set of points on the unit sphere.
    The covering radius is the maximum over all sphere points of the angular distance
    to the nearest configuration point. This equals the largest circumradius of the
    convex hull facets (each facet's circumcenter, if it lies on the sphere side,
    gives the worst-case point for that facet).
    
    For N=130, we use a practical approach: sample many points on the sphere and
    also check the convex hull facet circumcenters.
    """
    from scipy.spatial import ConvexHull
    
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Method 1: Check convex hull facets
    # The covering radius is the max angular distance from any sphere point to nearest pt.
    # The worst points are at the circumcenters of the Delaunay triangulation on sphere,
    # which correspond to facets of the convex hull.
    
    try:
        hull = ConvexHull(pts)
        # For each facet (triangle on the convex hull), compute the circumcenter of the
        # three points projected onto the sphere. The circumcenter of a spherical triangle
        # is the point equidistant from all three vertices on the sphere.
        
        max_ang = 0.0
        
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # The circumcenter of a spherical triangle: the point on the sphere
            # equidistant from a, b, c. This is the normalized cross product of
            # (b-a) and (c-a) ... actually, for the spherical circumcenter, we want
            # the point p on the sphere such that |p-a|=|p-b|=|p-c| (angular distances).
            # This is equivalent to p being orthogonal to (a x b), (b x c), (c x a) ... 
            # Actually, the spherical circumcenter is the normalized vector:
            # p = (a x b) x (b x c) ... no.
            # 
            # The spherical circumcenter is the unit vector in the direction of:
            #   (a - (a·b)b/|b|^2) ... 
            # 
            # Simpler: the circumcenter of triangle abc on the sphere is the point p
            # such that p·a = p·b = p·c (equal angular distance). So p is in the null
            # space of the matrix with rows [a-b, b-c] (as constraints on p).
            # p is orthogonal to (a-b) and (b-c), so p ∝ (a-b) x (b-c) = a x b - a x c - b x a + b x c
            # = a x b + b x c + c x a (since a x b = -b x a etc.)
            # Wait: (a-b) x (b-c) = a x b - a x c - b x b + b x c = a x b - a x c + b x c
            # And a x b + b x c + c x a = a x b + b x c - a x c. Yes, same thing.
            
            v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            nv = np.linalg.norm(v)
            if nv < 1e-14:
                # Degenerate triangle
                continue
            p = v / nv
            
            # The circumcenter could be on either side; the one on the "outside" of the hull
            # is the one we want. The angular distance from p to a is the circumradius.
            # But p might be the "far" circumcenter; we need the one that's a local maximum
            # of the distance function, which is the one opposite to the hull interior.
            # For a convex hull facet, the circumcenter that matters is the one on the
            # side of the facet away from the origin (i.e., p should have positive dot with
            # the facet normal pointing outward).
            
            # Actually, both p and -p are equidistant from a,b,c. The covering radius
            # uses the one that is farthest from all points, which is the one with
            # larger angular distance. Let's compute the angular distance for p.
            cos_d = np.clip(np.dot(p, a), -1.0, 1.0)
            ang = np.degrees(np.arccos(cos_d))
            
            # Also check -p
            cos_d2 = np.clip(np.dot(-p, a), -1.0, 1.0)
            ang2 = np.degrees(np.arccos(cos_d2))
            
            max_ang = max(max_ang, ang, ang2)
        
    except Exception:
        max_ang = 0.0
    
    # Method 2: Dense sampling to verify
    # Use a fine Fibonacci spiral as test points
    n_test = 50000
    i_t = np.arange(n_test) + 0.5
    z_t = 1.0 - 2.0 * i_t / n_test
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * i_t
    r_t = np.sqrt(np.maximum(0.0, 1.0 - z_t * z_t))
    test_pts = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    # Compute angular distance from each test point to nearest configuration point
    # Using dot products
    dots = test_pts @ pts.T  # (n_test, N)
    max_dots = np.max(dots, axis=1)  # closest point gives max dot (min angle)
    min_cos = 1.0 - max_dots  # 1 - cos(theta) for the nearest
    # theta = arccos(max_dot)
    angles = np.degrees(np.arccos(np.clip(max_dots, -1.0, 1.0)))
    sample_max = np.max(angles)
    
    return max(max_ang, sample_max)


def _repulsion_energy(pts):
    """Compute a repulsion-based energy that we want to minimize."""
    # Use a combination of: maximize minimum pairwise distance, and minimize covering radius proxy
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    # Angular distances
    cos_d = np.clip(dots, -1.0, 1.0)
    # We want to maximize the minimum angular distance between any two points
    # And minimize the covering radius
    # Energy: -min_pairwise_angle + covering_radius
    # But this is non-smooth. Instead, use a smooth approximation.
    
    # For optimization, use: sum of exp(-k * angular_distance) for all pairs
    # This pushes points apart.
    k = 5.0
    off_diag = cos_d - np.eye(N)
    # Angular distance
    angles_rad = np.arccos(np.clip(cos_d, -1.0, 1.0))
    # Zero out diagonal
    angles_rad = angles_rad * (1 - np.eye(N))
    
    # Energy: sum over all pairs of exp(-k * angle)
    # This is large when points are close
    energy_pairs = np.sum(np.exp(-k * angles_rad))
    
    return energy_pairs


def _optimize_repulsion(pts, n_iter=2000, lr=0.01, deadline=None):
    """
    Optimize points to maximize repulsion (minimize covering radius proxy).
    Uses gradient descent on the sphere with a repulsive potential.
    """
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Use a smooth energy: for each pair, energy = 1/(1 + exp(k*(angle - target)))
    # Or simpler: minimize sum of (target - angle)^2 for angle < target
    # Let's use: minimize -sum(log(sin(angle/2))) which maximizes product of sin(angle/2)
    # This is the log-likelihood of a uniform distribution on the sphere given the points.
    
    k = 8.0  # sharpness
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Compute gradient of repulsion energy
        # Energy = sum_{i<j} f(theta_ij) where theta_ij is angular distance
        # f(theta) = exp(-k*theta)
        # dE/dtheta_ij = -k * exp(-k*theta_ij)
        # dtheta_ij/dp_i = ... 
        
        # Alternative: use the fact that for points on sphere, we can use
        # the gradient of the log-determinant or similar.
        
        # Simpler approach: use a force-based method
        # Force on point i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^3 * strength
        # This is like Coulomb repulsion in 3D, projected onto the sphere.
        
        # Compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dists, 1.0)
        
        # Force: sum_j (p_i - p_j) / dist_ij^2 (in 3D)
        # This is attractive to spread out
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Repulsive force: direction from j to i, magnitude ~ 1/d^2
                # But we want to maximize minimum distance, so use a steeper potential
                force = (pts[i] - pts[j]) / (d * d * d)  # 1/d^3 for 1/d^2 potential
                forces[i] += force
        
        # Project forces onto tangent plane of sphere
        # Tangent projection: f_tan = f - (f·p_i) p_i
        for i in range(N):
            f = forces[i]
            f_tan = f - np.dot(f, pts[i]) * pts[i]
            pts[i] += lr * f_tan
        
        # Normalize back to sphere
        pts = _normalize(pts)
        
        # Periodically check covering radius
        if it % 50 == 0:
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    return best_pts, best_cr


def _local_search(pts, n_iterations=500, deadline=None):
    """
    Local search: try moving individual points to reduce covering radius.
    """
    pts = _normalize(pts).copy()
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Try to improve by moving each point
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 15:
                break
            
            # Find the current worst-case region for point i
            # Move point i in the direction that reduces the max angular distance
            # in its neighborhood
            
            # Get neighbors
            dots = pts @ pts[i]
            # The point i is responsible for the region where it's the nearest
            # Move it toward the centroid of its Voronoi cell (on the sphere)
            
            # Simple heuristic: move toward the average of nearby points' midpoints
            # Or: compute the gradient of the covering radius w.r.t. point i
            
            # For each test point in the neighborhood, compute if moving i helps
            # Use a few trial directions
            
            # Random perturbation
            rng = np.random.default_rng(it * N + i)
            for trial in range(3):
                if deadline is not None and time.time() > deadline - 15:
                    break
                
                # Generate a random tangent direction
                v = rng.standard_normal(3)
                v = v - np.dot(v, pts[i]) * pts[i]
                nv = np.linalg.norm(v)
                if nv < 1e-10:
                    continue
                v /= nv
                
                # Try a few step sizes
                for step in [0.05, 0.1, 0.2]:
                    new_pts = pts.copy()
                    new_pts[i] = _normalize(new_pts[i] + step * v)[0]
                    
                    cr = _covering_radius(new_pts)
                    if cr < best_cr - 1e-6:
                        best_cr = cr
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try larger random restarts for a few points
            for i in range(min(5, N)):
                if deadline is not None and time.time() > deadline - 15:
                    break
                rng = np.random.default_rng(1000 + it * N + i)
                v = rng.standard_normal(3)
                v = v - np.dot(v, pts[i]) * pts[i]
                nv = np.linalg.norm(v)
                if nv < 1e-10:
                    continue
                v /= nv
                for step in [0.3, 0.5, 0.7]:
                    new_pts = pts.copy()
                    new_pts[i] = _normalize(new_pts[i] + step * v)[0]
                    cr = _covering_radius(new_pts)
                    if cr < best_cr - 1e-6:
                        best_cr = cr
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                        break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # First, do repulsion-based optimization
    pts, cr = _optimize_repulsion(pts, n_iter=1000, lr=0.02, deadline=deadline)
    
    # Then, do local search
    if time.time() < deadline - 20:
        pts, cr = _local_search(pts, n_iterations=200, deadline=deadline)
    
    # Final normalization
    pts = _normalize(pts)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
        pts = _normalize(pts)
    
    return pts
