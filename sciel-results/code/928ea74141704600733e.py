import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130

def _covering_radius_from_hull(points):
    """
    Compute the covering radius from the convex hull facets.
    The covering radius is the maximum circumradius of the spherical caps
    defined by the facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        # Get the 3 points of this triangle
        tri = points[simplex]
        
        # Compute the circumradius of the spherical triangle.
        # For a spherical triangle with vertices on the unit sphere,
        # the circumradius (angular distance from circumcenter to each vertex)
        # can be computed from the side lengths.
        
        # Side lengths (angular distances)
        a = np.arccos(np.clip(np.dot(tri[1], tri[2]), -1, 1))
        b = np.arccos(np.clip(np.dot(tri[0], tri[2]), -1, 1))
        c = np.arccos(np.clip(np.dot(tri[0], tri[1]), -1, 1))
        
        # Spherical circumradius formula:
        # R = arcsin( (sin(a) * sin(b) * sin(c)) / (2 * |cos(a)cos(b)cos(c) + cos(a) + cos(b) + cos(c)|) )
        # But this is tricky. Alternative: find the circumcenter on the sphere.
        
        # The circumcenter of a spherical triangle is the point on the sphere
        # equidistant from all three vertices. It lies along the normal to the plane
        # containing the three points, projected onto the sphere.
        
        # Normal to the plane of the triangle
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-12:
            # Degenerate triangle
            continue
        
        normal /= norm_len
        
        # The circumcenter is either normal or -normal, whichever is closer
        # to the triangle (i.e., the one that gives a valid spherical triangle).
        # We need the one that is on the same side as the origin's opposite,
        # i.e., the one that makes the cap contain the triangle.
        
        # Actually, for the convex hull, the facet's outward normal points away
        # from the origin. The circumcenter of the spherical cap is along this normal.
        # The angular distance from the circumcenter to each vertex is the circumradius.
        
        # Try both directions
        for sign in [1, -1]:
            center = sign * normal
            # Check if this center is valid: the angular distances to all 3 vertices
            # should be <= pi/2 (center is within the hemisphere defined by the triangle)
            d0 = np.arccos(np.clip(np.dot(center, tri[0]), -1, 1))
            d1 = np.arccos(np.clip(np.dot(center, tri[1]), -1, 1))
            d2 = np.arccos(np.clip(np.dot(center, tri[2]), -1, 1))
            
            # For the circumcenter to be valid, all distances should be equal
            # (within numerical tolerance) and the center should be the "top" of the cap.
            if abs(d0 - d1) < 1e-8 and abs(d1 - d2) < 1e-8:
                # This is a valid circumcenter
                r = d0
                if r > max_radius:
                    max_radius = r
                break
        else:
            # If neither direction gave a valid circumcenter, use the plane-based approximation
            # The circumradius can be approximated by the distance from the triangle's
            # centroid to its vertices, but that's not exact. Let's use a fallback.
            # Actually, the normal direction should always work for a convex hull facet.
            # Let's just take the normal direction.
            center = normal
            d0 = np.arccos(np.clip(np.dot(center, tri[0]), -1, 1))
            d1 = np.arccos(np.clip(np.dot(center, tri[1]), -1, 1))
            d2 = np.arccos(np.clip(np.dot(center, tri[2]), -1, 1))
            # Use the maximum as a conservative estimate
            r = max(d0, d1, d2)
            if r > max_radius:
                max_radius = r
    
    return max_radius

def _soft_covering_radius_and_grad(points, n_grid=5000, beta=50.0):
    """
    Compute a soft approximation of the covering radius and its gradient.
    Uses a grid of points on the sphere and a soft-max/soft-min.
    """
    # Generate Fibonacci spiral grid
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Compute angular distances from each grid point to each configuration point
    # dot = grid_points @ points.T  (n_grid, N)
    dot = grid_points @ points.T
    # Clip for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    # Angular distance
    angular_dists = np.arccos(dot)  # (n_grid, N)
    
    # For each grid point, find the minimum angular distance to any configuration point
    # Use soft-min: -log-sum-exp(-beta * dists) / beta
    # soft_min = -(1/beta) * log(sum(exp(-beta * dists)))
    exp_neg_beta_dists = np.exp(-beta * angular_dists)  # (n_grid, N)
    log_sum_exp = np.log(np.sum(exp_neg_beta_dists, axis=1))  # (n_grid,)
    soft_min_dists = -log_sum_exp / beta  # (n_grid,)
    
    # Soft-max over grid points: (1/beta) * log(sum(exp(beta * soft_min_dists)))
    exp_beta_soft_min = np.exp(beta * soft_min_dists)
    log_sum_exp2 = np.log(np.sum(exp_beta_soft_min))
    soft_covering_radius = log_sum_exp2 / beta
    
    # Gradient with respect to points
    # d(soft_covering_radius)/d(points)
    # This is complex. Let's use a simpler approach:
    # The gradient of the soft-max of soft-min can be computed via chain rule.
    
    # Actually, for optimization, let's use a different approach:
    # Minimize the sum of squared distances to the nearest point, weighted by importance.
    # Or use the Lloyd's algorithm approach.
    
    # For now, return the soft covering radius and a zero gradient placeholder.
    # We'll use a different optimization strategy.
    grad = np.zeros_like(points)
    
    return soft_covering_radius, grad

def _lloyd_step(points, n_iter=50):
    """
    Lloyd's algorithm for spherical k-means.
    Each point is moved to the centroid (on the sphere) of its Voronoi cell.
    """
    for _ in range(n_iter):
        # Compute Voronoi cells using Delaunay triangulation
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(N):
            # Find all triangles that include point i
            # This is expensive. Alternative: use the convex hull approach.
            # For each point, find the centroid of its Voronoi cell.
            # The Voronoi cell of point i is the set of points on the sphere
            # closer to i than to any other point.
            
            # Approximate by finding the centroid of the nearest neighbors
            # or by using the normal of the convex hull facet.
            
            # Better: use the fact that in Lloyd's algorithm on the sphere,
            # the new position is the normalized sum of all grid points in the Voronoi cell.
            
            # For efficiency, let's use a simpler heuristic:
            # Move each point away from its nearest neighbors.
            pass
        
        # Simplified Lloyd step: for each point, compute the weighted average
        # of nearby points and move towards the "center" of the empty space.
        
        # Actually, let's use a different approach: gradient descent on the
        # sum of squared distances to the nearest point.
        
        # Compute nearest neighbor distance for each point
        # and move points to reduce the maximum distance.
        
        # For now, break and use a different optimization.
        break
    
    return points

def _exact_local_optimize(points, time_limit=10.0):
    """
    Local optimization using L-BFGS-B on the soft covering radius.
    """
    start_time = time.time()
    
    # Flatten points
    x0 = points.flatten()
    
    def objective(x_flat):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        val, _ = _soft_covering_radius_and_grad(pts, n_grid=3000, beta=30.0)
        return val
    
    # Provide a simple gradient by finite differences? No, too slow.
    # Use L-BFGS-B with numerical gradient.
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if res.success or res.status in [1, 2]:
            opt_points = res.x.reshape(N, 3)
            norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            opt_points = opt_points / norms
            return opt_points
        else:
            return points
    except Exception:
        return points

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(spiral_points)
    
    # 2. Random uniform
    rng = np.random.default_rng(42)
    for _ in range(3):
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # 3. Icosahedron-based with jitter
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    # Expand to 130 points by adding jittered copies
    ico_expanded = []
    for v in ico_verts:
        for _ in range(10):
            jittered = v + rng.normal(scale=0.1, size=3)
            jittered /= np.linalg.norm(jittered)
            ico_expanded.append(jittered)
    ico_expanded = np.array(ico_expanded[:N])
    starts.append(ico_expanded)
    
    # Evaluate each start and optimize
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 15.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius_from_hull(start_pts)
        
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Local optimization
        remaining = total_time_limit - (time.time() - start_time)
        n_left = len(starts) - idx
        time_per_opt = max(5.0, min(15.0, (remaining - 10.0) / max(1, n_left)))
        
        opt_points = _exact_local_optimize(start_pts, time_limit=time_per_opt)
        
        # Re-evaluate
        score = _covering_radius_from_hull(opt_points)
        
        if score < best_score:
            best_score = score
            best_points = opt_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
