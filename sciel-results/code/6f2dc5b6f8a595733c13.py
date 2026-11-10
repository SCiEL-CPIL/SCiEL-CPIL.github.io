"""Improved covering-radius solver for 130 points on the unit sphere."""
import time
import numpy as np

N = 130
_TIME_LIMIT = 115.0  # safety margin under 120s / 150s kills


def _covering_radius(points: np.ndarray) -> float:
    """Compute covering radius in radians using convex hull facets."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # invalid -> worst possible
    # For each facet (simplex), the farthest point on the sphere from the
    # nearest configuration point is the point opposite the facet's outward normal.
    # The covering radius is the max over facets of the angle between the facet
    # normal (pointing outward from origin) and the nearest point.
    # Actually: for a facet with vertices v0,...,vk, the point on the sphere farthest
    # from all points is the one in direction of the facet's outward normal n.
    # The distance from n to the nearest point is what matters.
    # But more precisely: the covering radius = max over facets of the circumradius
    # of that facet (angular). For a simplex facet, the circumcenter on the sphere
    # is along the normalized normal. The angular distance from that circumcenter
    # to any vertex of the facet is the facet's circumradius.
    
    # Get facet normals
    normals = []
    for simplex in hull.simplices:
        verts = points[simplex]
        # Compute normal via cross product of two edges (for triangles)
        # For higher-dim simplices, use the fact that in 3D hull facets are triangles
        # (or we can compute via SVD)
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            n = np.cross(e1, e2)
            n = n / np.linalg.norm(n)
            # Ensure outward: normal should point away from origin
            if np.dot(n, verts[0]) < 0:
                n = -n
            normals.append(n)
        else:
            # Fallback: use centroid direction
            c = np.mean(verts, axis=0)
            n = c / np.linalg.norm(c)
            normals.append(n)
    
    normals = np.array(normals)
    
    # For each facet normal, find the nearest point and compute angular distance
    # The covering radius is the max over all facets of the max angular distance
    # from the facet's circumcenter (which is along the normal) to the nearest point.
    # Actually, the covering radius is the maximum, over all directions on the sphere,
    # of the angular distance to the nearest point. This equals the maximum over
    # facets of the angular circumradius of that facet.
    
    # For a triangular facet with vertices a, b, c, the circumradius (angular) is
    # the angle between the circumcenter (normalized normal) and any vertex.
    # But we need the MAX over all sphere points, which is the circumcenter of the
    # "empty" cap. The covering radius = max over facets of angle(n, nearest point).
    
    # Compute for each normal the max angle to nearest point
    # Actually: the covering radius = max over all y on sphere of min_i angle(y, p_i)
    # This equals max over facets of the circumradius of the facet (in angular terms)
    # where the circumcenter is the point on the sphere farthest from all p_i within
    # that facet's Voronoi cell.
    
    # Simpler: for each facet normal n, the angular distance to the nearest point
    # gives a lower bound. The actual covering radius is the max over all facets.
    
    # Let's compute: for each facet, the circumcenter direction is n.
    # The angular distance from n to the nearest point p_i is arccos(n . p_i).
    # The covering radius is the max over facets of this value.
    
    # Compute dot products: normals @ points.T
    dots = normals @ points.T  # shape (n_facets, N)
    # Clip for numerical safety
    dots = np.clip(dots, -1.0, 1.0)
    # For each facet, find min dot product (max angle)
    min_dots = np.min(dots, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    return np.max(angles)


def _smooth_covering_radius(points: np.ndarray, beta: float = 50.0) -> float:
    """Smooth approximation of covering radius using log-sum-exp on random samples."""
    # Sample points on the sphere
    n_samples = 2000
    # Use deterministic random for reproducibility
    rng = np.random.RandomState(42)
    # Generate random points on sphere
    u = rng.rand(n_samples, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    
    # Compute angular distances from each sample to all points
    dots = u @ points.T  # (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_samples, N)
    
    # For each sample, find min angle
    min_angles = np.min(angles, axis=1)
    
    # Smooth max via log-sum-exp
    max_angle = np.max(min_angles)
    shifted = min_angles - max_angle
    log_sum_exp = max_angle + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp


def _optimize_smooth(points: np.ndarray, time_budget: float) -> np.ndarray:
    """Optimize using L-BFGS-B on smooth covering radius."""
    from scipy.optimize import minimize
    
    best_points = points.copy()
    best_val = _covering_radius(points)
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        return _smooth_covering_radius(pts, beta=80.0)
    
    x0 = points.flatten()
    
    start = time.time()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 20000})
    
    if result.success or result.fun < best_val:
        pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        val = _covering_radius(pts)
        if val < best_val:
            best_points = pts
            best_val = val
    
    elapsed = time.time() - start
    return best_points


def _local_refine(points: np.ndarray, time_budget: float) -> np.ndarray:
    """Local refinement: move each point to the circumcenter of its worst facet."""
    best_points = points.copy()
    best_val = _covering_radius(points)
    
    start = time.time()
    improved = True
    max_iter = 50
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        if not improved:
            break
        improved = False
        
        # Compute covering radius and identify worst facet
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(best_points)
        except Exception:
            break
        
        # Find the facet with largest circumradius
        worst_idx = -1
        worst_angle = 0.0
        for i, simplex in enumerate(hull.simplices):
            verts = best_points[simplex]
            if len(verts) == 3:
                e1 = verts[1] - verts[0]
                e2 = verts[2] - verts[0]
                n = np.cross(e1, e2)
                n = n / np.linalg.norm(n)
                if np.dot(n, verts[0]) < 0:
                    n = -n
                # Find nearest point to n
                dots = best_points @ n
                min_dot = np.min(dots)
                angle = np.arccos(np.clip(min_dot, -1.0, 1.0))
                if angle > worst_angle:
                    worst_angle = angle
                    worst_idx = i
        
        if worst_idx < 0:
            break
        
        simplex = hull.simplices[worst_idx]
        verts = best_points[simplex]
        
        # Compute circumcenter of the facet (on the sphere)
        # For a triangle on the unit sphere, the circumcenter is the normalized
        # vector that is equidistant from all three vertices.
        # Solve: c . a = c . b = c . c, |c| = 1
        # This means c is perpendicular to (a-b) and (b-c)
        a, b, c = verts
        # c is in the plane spanned by the normal and the incenter direction
        # Use: circumcenter = normalize(a + b + c) doesn't work generally
        # Better: solve linear system
        # c . (a - b) = 0, c . (b - c) = 0, c . c = 1
        # c = alpha * n + beta * m where n is normal, m is in-plane
        # Simpler: the circumcenter of a spherical triangle is the intersection
        # of the sphere with the line through the origin in the direction of
        # the circumcenter of the planar triangle formed by projecting to a plane.
        
        # Use the formula: circumcenter direction = (a×b + b×c + c×a) normalized
        # This is the direction of the vector equidistant from a, b, c on the sphere
        # Actually, the circumcenter of a spherical triangle is given by:
        # c_dir = (sin(A)*a + sin(B)*b + sin(C)*c) / |...| where A,B,C are angles
        # But simpler: c is perpendicular to (a-b) and (b-c)
        # So c = normalize( (a×b)×(b×c) ) ... no
        
        # Let's just use: the point on the sphere equidistant from a, b, c
        # satisfies c·a = c·b = c·c
        # So c·(a-b) = 0 and c·(b-c) = 0
        # c is in the null space of [[a-b], [b-c]]
        # c = normalize( cross(a-b, b-c) )
        d1 = a - b
        d2 = b - c
        c_dir = np.cross(d1, d2)
        c_norm = np.linalg.norm(c_dir)
        if c_norm < 1e-12:
            # Degenerate, use centroid
            c_dir = (a + b + c) / 3.0
            c_dir = c_dir / np.linalg.norm(c_dir)
        else:
            c_dir = c_dir / c_norm
        
        # Move one of the points toward the circumcenter to reduce the gap
        # Choose the point that is farthest from the circumcenter among the facet vertices
        # Actually, we should move a point that is "responsible" for the gap
        # Better approach: add a new point at the circumcenter and remove the worst point
        # But we need to keep N points. Instead, move the point that is nearest to
        # the circumcenter direction but still creates the large gap.
        
        # Alternative: gradient step - move all points slightly to reduce smooth covering radius
        # Let's do a small perturbation: move the point closest to the circumcenter
        # in the direction of the circumcenter
        
        # Find which point is closest to c_dir
        dots = best_points @ c_dir
        closest_idx = np.argmax(dots)  # closest to c_dir
        
        # Move it slightly toward c_dir
        step = 0.01
        new_point = best_points[closest_idx] + step * (c_dir - best_points[closest_idx])
        new_point = new_point / np.linalg.norm(new_point)
        
        trial = best_points.copy()
        trial[closest_idx] = new_point
        trial_val = _covering_radius(trial)
        
        if trial_val < best_val - 1e-12:
            best_points = trial
            best_val = trial_val
            improved = True
    
    return best_points


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_val = _covering_radius(points)
    
    # Phase 1: Smooth optimization with L-BFGS
    remaining = _TIME_LIMIT - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        points = _optimize_smooth(points, remaining * 0.6)
        val = _covering_radius(points)
        if val < best_val:
            best_points = points
            best_val = val
    
    # Phase 2: Local refinement
    remaining = _TIME_LIMIT - (time.time() - start_time) - 3.0
    if remaining > 5.0:
        points = _local_refine(best_points, remaining * 0.8)
        val = _covering_radius(points)
        if val < best_val:
            best_points = points
            best_val = val
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
