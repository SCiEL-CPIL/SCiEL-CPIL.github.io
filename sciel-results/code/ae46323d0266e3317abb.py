"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE_FRACTION = 0.90  # safety margin before 120s wall-clock


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _angular_distance_deg(pts, y):
    """Angular distance in degrees from y (unit vector) to each point in pts."""
    dots = np.clip(pts @ y, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _covering_radius_deg(pts):
    """
    Covering radius: max over facets of the convex hull of the circumradius
    (angular distance from facet center to its vertices), but only for facets
    whose cap is empty (i.e., all points are on one side).
    For a set of points on the unit sphere, the covering radius equals the
    maximum circumradius of the Delaunay triangulation on the sphere.
    We approximate using the convex hull: for each face, compute the circumcenter
    and the max angular distance to the face vertices.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max nearest-neighbor distance on a dense grid
        return _covering_radius_grid(pts)
    
    faces = hull.simplices
    max_cr = 0.0
    for face in faces:
        v = pts[face]  # shape (3, 3) typically
        # Circumcenter of the triangle on the unit sphere
        # The circumcenter direction is the normalized cross product approach
        a, b, c = v[0], v[1], v[2]
        # Normal to the face
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        
        # Check if the cap is empty: all points should be on the same side
        # (the side opposite to the origin, i.e., dot with n <= 0 for interior)
        # For a convex hull of points on the unit sphere, the outward normal
        # points away from the origin. The circumcenter of the spherical triangle
        # is in the direction of n if the triangle is small enough.
        
        # Compute circumcenter: it's the point on the sphere equidistant from a, b, c
        # Solve: x·a = x·b = x·c = cos(theta), |x|=1
        # This gives a linear system: (a-b)·x = 0, (a-c)·x = 0, and x·n = cos(theta)
        # Actually, the circumcenter lies along the normal to the plane through a,b,c
        # only if the triangle is acute. For obtuse triangles, the circumcenter
        # is at the midpoint of the longest edge.
        
        # Let's compute the circumcenter properly.
        # The circumcenter x satisfies: x·a = x·b = x·c
        # So (a-b)·x = 0 and (a-c)·x = 0
        # Thus x is parallel to (a-b) × (a-c) = normal direction
        # But we also need |x| = 1.
        
        # If the triangle is acute (all angles < 90°), the circumcenter is in the
        # direction of the outward normal. Otherwise, it's at the midpoint of the
        # longest edge.
        
        # Check angles:
        # Angle at a: cos(A) = (b-a)·(c-a) / (|b-a||c-a|)
        ab = b - a
        ac = c - a
        bc = c - b
        ba = a - b
        ca = a - c
        cb = b - c
        
        dot_ab_ac = np.dot(ab, ac)
        dot_ba_bc = np.dot(ba, bc)
        dot_ca_cb = np.dot(ca, cb)
        
        norm_ab = np.linalg.norm(ab)
        norm_ac = np.linalg.norm(ac)
        norm_bc = np.linalg.norm(bc)
        
        cos_A = dot_ab_ac / (norm_ab * norm_ac + 1e-12)
        cos_B = dot_ba_bc / (norm_ab * norm_bc + 1e-12)
        cos_C = dot_ca_cb / (norm_ac * norm_bc + 1e-12)
        
        # If all angles are acute, circumcenter is along normal
        if cos_A > 0 and cos_B > 0 and cos_C > 0:
            cc_dir = n
            # Determine sign: the circumcenter should be on the same side as the
            # points relative to the face plane. For convex hull outward normal,
            # the circumcenter of the spherical triangle is in the direction of n
            # if the triangle subtends less than 180°.
            # Check: dot(a, n) should be positive for the circumcenter to be valid
            # in this direction.
            if np.dot(a, n) < 0:
                cc_dir = -n
        else:
            # Obtuse: circumcenter is at the midpoint of the longest edge
            if norm_bc >= norm_ab and norm_bc >= norm_ac:
                cc_dir = (b + c)
            elif norm_ab >= norm_ac:
                cc_dir = (a + b)
            else:
                cc_dir = (a + c)
            cc_norm = np.linalg.norm(cc_dir)
            if cc_norm < 1e-12:
                continue
            cc_dir = cc_dir / cc_norm
        
        # The circumradius is the angular distance from cc_dir to any vertex
        dots = np.clip(np.array([np.dot(cc_dir, a), np.dot(cc_dir, b), np.dot(cc_dir, c)]), -1.0, 1.0)
        cr = np.degrees(np.arccos(dots)).max()
        max_cr = max(max_cr, cr)
    
    return max_cr


def _covering_radius_grid(pts, n_lat=60, n_lon=120):
    """Fallback: compute covering radius on a dense spherical grid."""
    lat = np.linspace(-np.pi/2 + np.pi/(2*n_lat), np.pi/2 - np.pi/(2*n_lat), n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    y = np.column_stack([
        np.cos(lat_grid) * np.cos(lon_grid).flatten(),
        np.cos(lat_grid) * np.sin(lon_grid).flatten(),
        np.sin(lat_grid).flatten()
    ])
    # For each grid point, find the nearest point
    dots = y @ pts.T  # shape (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    nearest = np.arccos(dots).min(axis=1)
    return np.degrees(nearest.max())


def _objective_and_grad(x, pts_flat):
    """
    Objective: smooth approximation of the covering radius using a soft-max
    over the distances from a set of test points to the nearest configuration point.
    We use a fixed set of test points on the sphere.
    """
    # Reshape
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Test points: use a moderate grid for speed
    n_lat = 30
    n_lon = 60
    lat = np.linspace(-np.pi/2 + np.pi/(2*n_lat), np.pi/2 - np.pi/(2*n_lat), n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    y = np.column_stack([
        np.cos(lat_grid) * np.cos(lon_grid).flatten(),
        np.cos(lat_grid) * np.sin(lon_grid).flatten(),
        np.sin(lat_grid).flatten()
    ])
    
    # For each test point, compute distance to nearest configuration point
    dots = y @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_test, N)
    min_dists = dists.min(axis=1)  # (n_test,)
    
    # Soft-max with temperature
    beta = 50.0
    log_sum_exp = np.log(np.sum(np.exp(-beta * min_dists)))
    soft_max = -np.log(np.sum(np.exp(-beta * min_dists))) / beta
    # Actually, soft-min: we want to minimize the max, so we use:
    # soft_min(d) = -(1/beta) * log(sum(exp(-beta * d)))
    # This is a smooth lower bound on the min... wait, we want the max of the min distances.
    # The covering radius is max_y min_p d(y,p).
    # So we have an array of min_dists (one per test point), and we want the max.
    # Soft-max: (1/beta) * log(sum(exp(beta * d)))
    soft_max = np.log(np.sum(np.exp(beta * min_dists))) / beta
    
    # Gradient: d(soft_max)/d(x)
    # soft_max = (1/beta) * log(sum_j exp(beta * d_j))
    # d(soft_max)/d_j = exp(beta * d_j) / sum_k exp(beta * d_k) = softmax_j
    # d_j = min_p d(y_j, p)
    # The gradient of d_j with respect to x is complicated because of the min.
    # We approximate by using the argmin point for each test point.
    
    # This is getting complex. Let's use a simpler approach: 
    # Just minimize the sum of exp(beta * d_j) which has similar gradient properties.
    
    return soft_max


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_FRACTION * 120.0
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_score = _covering_radius_grid(best_pts)
    
    # Try local optimization using gradient descent on the smooth surrogate
    # We'll use a simpler approach: iterative refinement by moving points
    # toward the circumcenters of their Voronoi facets.
    
    # Precompute test points for faster evaluation
    n_lat = 40
    n_lon = 80
    lat = np.linspace(-np.pi/2 + np.pi/(2*n_lat), np.pi/2 - np.pi/(2*n_lat), n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    test_pts = np.column_stack([
        np.cos(lat_grid) * np.cos(lon_grid).flatten(),
        np.cos(lat_grid) * np.sin(lon_grid).flatten(),
        np.sin(lat_grid).flatten()
    ])
    
    def _eval_score(pts):
        """Fast evaluation using test points."""
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots).min(axis=1)
        return np.degrees(dists.max())
    
    def _gradient_step(pts, step_size=0.01):
        """
        Move each point toward the centroid of its Voronoi cell (approximated
        by the test points for which it is the nearest neighbor).
        """
        dots = test_pts @ pts.T  # (n_test, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (n_test, N)
        nearest_idx = dists.argmin(axis=1)  # (n_test,)
        
        new_pts = pts.copy()
        for k in range(N):
            mask = (nearest_idx == k)
            if mask.sum() == 0:
                continue
            # Centroid of the test points in this Voronoi cell
            centroid = test_pts[mask].mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm < 1e-12:
                continue
            centroid = centroid / c_norm
            # Move toward the centroid
            new_pts[k] = pts[k] + step_size * (centroid - pts[k])
        
        return _normalize(new_pts)
    
    # Iterative refinement
    step_size = 0.05
    n_iter_max = 200
    for it in range(n_iter_max):
        if time.time() > deadline:
            break
        
        new_pts = _gradient_step(best_pts, step_size)
        new_score = _eval_score(new_pts)
        
        if new_score < best_score:
            best_pts = new_pts
            best_score = new_score
            # Decrease step size slowly
            step_size *= 0.995
        else:
            # Try smaller step
            step_size *= 0.9
            if step_size < 0.001:
                break
    
    # Final polish with higher resolution
    if time.time() < deadline - 10:
        n_lat2 = 60
        n_lon2 = 120
        lat2 = np.linspace(-np.pi/2 + np.pi/(2*n_lat2), np.pi/2 - np.pi/(2*n_lat2), n_lat2)
        lon2 = np.linspace(0, 2*np.pi, n_lon2, endpoint=False)
        lat_grid2, lon_grid2 = np.meshgrid(lat2, lon2, indexing='ij')
        test_pts2 = np.column_stack([
            np.cos(lat_grid2) * np.cos(lon_grid2).flatten(),
            np.cos(lat_grid2) * np.sin(lon_grid2).flatten(),
            np.sin(lat_grid2).flatten()
        ])
        
        def _eval_score2(pts):
            dots = test_pts2 @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots).min(axis=1)
            return np.degrees(dists.max())
        
        def _gradient_step2(pts, step_size=0.01):
            dots = test_pts2 @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            nearest_idx = dists.argmin(axis=1)
            
            new_pts = pts.copy()
            for k in range(N):
                mask = (nearest_idx == k)
                if mask.sum() == 0:
                    continue
                centroid = test_pts2[mask].mean(axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-12:
                    continue
                centroid = centroid / c_norm
                new_pts[k] = pts[k] + step_size * (centroid - pts[k])
            return _normalize(new_pts)
        
        step_size = 0.02
        for it in range(100):
            if time.time() > deadline:
                break
            new_pts = _gradient_step2(best_pts, step_size)
            new_score = _eval_score2(new_pts)
            if new_score < best_score:
                best_pts = new_pts
                best_score = new_score
                step_size *= 0.995
            else:
                step_size *= 0.9
                if step_size < 0.0005:
                    break
    
    # Verify with convex hull-based covering radius
    final_score = _covering_radius_deg(best_pts)
    
    return best_pts
