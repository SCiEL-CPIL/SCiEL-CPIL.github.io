"""Program for covering the sphere with 130 equal caps using Lloyd/Voronoi relaxation."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

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
            faces = hull.simplices  # shape (F, 3)
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

    def lloyd_iteration(points, alpha=1.0):
        """
        One Lloyd/Voronoi relaxation step:
        1. Compute the convex hull to find Voronoi cells (in the sense of spherical caps).
        2. For each point, compute the "centroid" of its Voronoi cell on the sphere.
           The Voronoi cell of point i is the set of sphere points closest to i.
           A practical approximation: move each point toward the centroid of its 
           adjacent hull vertices (neighbors in the triangulation).
        """
        try:
            hull = ConvexHull(points)
            faces = hull.simplices  # (F, 3)
            vertices = hull.vertices  # indices of points on hull
            
            # For each point, collect its neighbors via the faces
            # Build adjacency
            adj = [[] for _ in range(N)]
            for f in faces:
                for i in range(3):
                    j = (i + 1) % 3
                    a_idx = f[i]
                    b_idx = f[j]
                    adj[a_idx].append(b_idx)
                    adj[b_idx].append(a_idx)
            
            new_points = points.copy()
            for i in range(N):
                if len(adj[i]) > 0:
                    neighbors = points[adj[i]]
                    # Spherical mean (centroid) of neighbors + self
                    # Use the normalized sum of all points in the cell
                    # Approximate cell as: point i + its neighbors
                    cell_points = np.vstack([points[i], neighbors])
                    centroid = np.sum(cell_points, axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        centroid /= norm
                        # Damped update
                        new_points[i] = (1 - alpha) * points[i] + alpha * centroid
                        new_points[i] /= np.linalg.norm(new_points[i])
            return new_points
        except Exception:
            return points

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** exponent)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with varying exponent
    n_restarts = 3
    exponents = [1.5, 2.0, 2.5]

    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(100 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart]
        points = gradient_descent(points, 0.05, 0.995, 300, exp)
        
        # Lloyd iterations for Voronoi relaxation
        lloyd_iters = 200
        for it in range(lloyd_iters):
            if time.time() > deadline - 40.0:
                break
            points = lloyd_iteration(points, alpha=0.5)
        
        rad = compute_covering_radius(points)

        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with gradient descent
    if best_points is not None and time.time() < deadline - 25.0:
        for trial_alpha in [0.01, 0.005]:
            if time.time() > deadline - 10.0:
                break
            candidate = best_points.copy()
            candidate = gradient_descent(candidate, trial_alpha, 0.99, 200, 2.0)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: More Lloyd iterations for polishing
    if best_points is not None and time.time() < deadline - 15.0:
        candidate = best_points.copy()
        lloyd_iters = 300
        for it in range(lloyd_iters):
            if time.time() > deadline - 10.0:
                break
            candidate = lloyd_iteration(candidate, alpha=0.3)
        rad = compute_covering_radius(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()

    # Final polish with fine gradient descent
    if best_points is not None and time.time() < deadline - 3.0:
        final = gradient_descent(best_points.copy(), 0.001, 0.995, 100, 2.0)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points
