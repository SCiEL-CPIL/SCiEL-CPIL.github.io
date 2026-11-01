"""Program for covering the sphere with 130 equal caps using Voronoi/Lloyd's Algorithm."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def lloyd_iteration(points, step_size=0.5):
        """
        Perform one Lloyd's iteration on the sphere.
        1. Compute ConvexHull to get Voronoi cells (dual to hull faces).
        2. For each point, find the 'worst' point in its cell (vertex of the cell's polytope farthest from center).
        3. Move point towards that worst point.
        """
        try:
            hull = ConvexHull(points)
            faces = hull.simplices  # (F, 3)
            # hull.equations: (F, 4) -> [a, b, c, d] for ax+by+cz+d <= 0
            # The normal vector of the face is (a,b,c). This normal points OUTSIDE the hull.
            # The Voronoi cell of point i is defined by inequalities relative to its neighbors.
            
            # Map hull vertices to points indices
            # hull.simplices gives the points forming each face.
            # We need to group faces by the vertex they are incident to? 
            # Actually, in a Delaunay/Voronoi duality on the sphere:
            # The Voronoi cell of point p_i is the intersection of hemispheres defined by p_i and its neighbors.
            # A simpler approach for Lloyd on sphere:
            # The "worst point" for point i is the circumcenter of the triangle of neighbors that is farthest from i?
            # No, the covering radius is determined by the circumcenters of the Delaunay triangles.
            # So, for each Delaunay triangle (face of hull), compute its circumcenter (on sphere).
            # That circumcenter is the worst-case point for the 3 points defining the face.
            # Lloyd's step: Move each point towards the circumcenters of the faces it touches.
            
            a = points[faces[:, 0]]
            b = points[faces[:, 1]]
            c = points[faces[:, 2]]
            
            # Compute circumcenter of each face (triangle on sphere)
            # The circumcenter is in the plane of the triangle.
            # Normal to triangle: n = (b-a) x (c-a)
            # But we want the point on the sphere equidistant from a, b, c.
            # This is the intersection of the sphere with the line passing through the origin and the circumcenter of the triangle in 3D?
            # Yes, the circumcenter of a spherical triangle is the normalization of the 3D circumcenter of the triangle formed by a,b,c.
            
            ab = b - a
            ac = c - a
            n = np.cross(ab, ac)
            norms_n = np.linalg.norm(n, axis=1, keepdims=True)
            valid = norms_n > 1e-10
            norms_n[valid] = 1.0
            norms_n[~valid] = 1.0
            n = n / norms_n
            
            # Circumcenter in 3D:
            # C = a + alpha * (ab x ac) ... wait.
            # Formula for circumcenter of triangle with vertices a, b, c:
            # d = 2 * (a x b) . c
            # C = (|a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b)) / d
            # Since points are on unit sphere, |a|^2 = 1.
            
            ab_cross_bc = np.cross(a, b) # a x b
            bc_cross_ca = np.cross(b, c) # b x c
            ca_cross_ab = np.cross(c, a) # c x a
            
            # Note: a x b + b x c + c x a is NOT generally zero, but relates to area vector.
            # Let's use the standard formula:
            # C_unnorm = 1*(b x c) + 1*(c x a) + 1*(a x b)
            # Wait, the formula is:
            # C = ( |b|^2 (c x a) + |c|^2 (a x b) + |a|^2 (b x c) ) / ( 2 * (a x b) . c )
            # Since |a|=|b|=|c|=1:
            # Numerator = (c x a) + (a x b) + (b x c)
            # Denominator = 2 * (a x b) . c
            
            num = (c - a) # placeholder, will compute properly
            # Vectorized:
            cross_bc = np.cross(b, c)
            cross_ca = np.cross(c, a)
            cross_ab = np.cross(a, b)
            
            numerator = cross_bc + cross_ca + cross_ab
            denominator = 2.0 * np.sum(cross_ab * c, axis=1, keepdims=True)
            
            # Handle degeneracy
            valid_den = np.abs(denominator) > 1e-10
            denominator[valid_den] = 1.0
            denominator[~valid_den] = 1.0
            
            circum_centers_3d = numerator / denominator
            
            # Normalize to sphere
            norms_cc = np.linalg.norm(circum_centers_3d, axis=1, keepdims=True)
            valid_cc = norms_cc > 1e-10
            norms_cc[valid_cc] = 1.0
            norms_cc[~valid_cc] = 1.0
            circum_centers = circum_centers_3d / norms_cc
            
            # For each point, collect the circumcenters of all faces it is part of.
            # We want to move the point towards the "average" of these circumcenters?
            # Or towards the one that is farthest?
            # Lloyd's algorithm for k-means moves to the centroid of the cluster.
            # For covering, we want to minimize the max distance.
            # The "worst" point in the cell is one of the circumcenters.
            # Specifically, the covering radius of the configuration is the max over all circumcenters of the distance to the nearest site.
            # For a specific site i, its local covering radius is the max distance from i to any circumcenter of a face containing i.
            # To minimize this max, we should move i towards the circumcenter that is currently farthest from i.
            # Let's compute the distance from each point to each circumcenter of its incident faces.
            
            # Build incidence: point i is in face f if i is in faces[f].
            # We can use an array of lists or just loop.
            # Since N=130, F ~ 2N-4 = 256. Looping is fast.
            
            new_points = points.copy()
            for i in range(N):
                # Find all faces containing point i
                # faces is (F, 3). Check if i is in faces[:, 0], faces[:, 1], or faces[:, 2]
                mask = (faces[:, 0] == i) | (faces[:, 1] == i) | (faces[:, 2] == i)
                if not np.any(mask):
                    continue
                
                incident_ccs = circum_centers[mask]
                if len(incident_ccs) == 0:
                    continue
                
                # Compute distances from points[i] to these circumcenters
                p = points[i]
                dots = np.clip(np.sum(incident_ccs * p, axis=1), -1.0, 1.0)
                dists = np.arccos(dots)
                
                # Find the farthest circumcenter
                max_idx = np.argmax(dists)
                target = incident_ccs[max_idx]
                
                # Move towards target
                # Geodesic step
                dot_pt = np.clip(np.sum(p * target, axis=1) if target.ndim > 1 else np.sum(p * target), -1.0, 1.0)
                if np.abs(dot_pt) < 1.0:
                    # Rotate p towards target
                    # v = target - (target . p) p
                    proj = np.sum(target * p)
                    v = target - proj * p
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-10:
                        v = v / norm_v
                        # Rotate by angle theta
                        # We want to reduce the distance.
                        # Let's just do a simple vector addition and renormalize, scaled by step_size
                        new_points[i] = p + step_size * v
                        new_points[i] /= np.linalg.norm(new_points[i])
            
            return new_points

        except Exception:
            return points

    def polish_lloyd(points, n_iter=20, step_size=0.1):
        """Refine with smaller steps."""
        for i in range(n_iter):
            points = lloyd_iteration(points, step_size)
        return points

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with Lloyd's
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Perturbed fibonacci
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Random start with different seed
            rng = np.random.RandomState(200)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Use best found so far with perturbation
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.2 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Run Lloyd's iterations
        # Start with larger steps, then smaller
        for i in range(50):
            if time.time() > deadline - 20.0:
                break
            step = 0.5 * (0.95 ** i)
            points = lloyd_iteration(points, step)
        
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Polish the best result
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        # Final polish with very small steps
        for i in range(50):
            if time.time() > deadline - 5.0:
                break
            points = lloyd_iteration(points, 0.05)
        
        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
