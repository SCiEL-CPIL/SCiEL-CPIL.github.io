"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points(n):
        pts = np.random.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def compute_face_radii(pts):
        """Return array of face circumradii (in radians) and faces."""
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

            return radii, faces
        except Exception:
            return None, None

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        radii, faces = compute_face_radii(pts)
        if radii is not None:
            return float(np.max(radii))
        # Fallback: grid-based estimate
        lat = np.linspace(-np.pi/2, np.pi/2, 80)
        lon = np.linspace(0, 2*np.pi, 160, endpoint=False)
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

    def lloyd_iteration(pts, n_iter=100):
        """Lloyd-style Voronoi centroid refinement on the sphere."""
        pts = pts.copy()
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            # Compute Voronoi cell centroids using spherical geometry
            # For each point, find the region where it's the nearest neighbor
            # Approximate by computing the centroid of points in its Voronoi cell
            # Use a grid-based approach for robustness
            lat = np.linspace(-np.pi/2, np.pi/2, 40)
            lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

            # Assign each grid point to nearest configuration point
            dot = np.dot(grid_pts, pts.T)
            nearest = np.argmax(dot, axis=1)

            # Compute centroids of each Voronoi cell
            new_pts = np.zeros_like(pts)
            counts = np.zeros(N)
            for i in range(N):
                mask = nearest == i
                if np.any(mask):
                    new_pts[i] = np.sum(grid_pts[mask], axis=0)
                    counts[i] = np.sum(mask)
                else:
                    new_pts[i] = pts[i]

            # Normalize
            new_pts = normalize(new_pts)
            pts = new_pts
        return pts

    def repulsion_step(points, alpha):
        """Standard repulsion step with inverse-cube weighting."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        return normalize(points)

    def repulsion_step_density_aware(points, alpha, face_radii=None, faces=None):
        """Repulsion with density-aware weighting."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)

        if face_radii is not None and faces is not None:
            vertex_max_rad = np.zeros(len(points))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    vertex_max_rad[vi] = max(vertex_max_rad[vi], face_radii[fi])
            if np.max(vertex_max_rad) > 1e-10:
                weight_boost = 1.0 + 0.3 * vertex_max_rad / np.max(vertex_max_rad)
                w = w * weight_boost[:, np.newaxis]

        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        return normalize(points)

    def optimize_worst_facet(pts, face_idx, faces, max_iter=50):
        """Try to optimize a specific worst facet by moving its vertices."""
        if face_idx is None or face_idx >= len(faces):
            return pts, False

        f = faces[face_idx]
        v0, v1, v2 = f

        # Get current positions
        p0 = pts[v0].copy()
        p1 = pts[v1].copy()
        p2 = pts[v2].copy()

        # Compute circumcenter direction
        a, b, c = p0, p1, p2
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        c_dir = ab + bc + ca
        norm_c = np.linalg.norm(c_dir)
        if norm_c < 1e-10:
            return pts, False
        c_dir = c_dir / norm_c

        # The circumradius is the angle from any vertex to circumcenter
        cos_r = np.clip(np.dot(c_dir, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > np.pi / 2:
            r = np.pi - r

        # Try moving vertices toward the circumcenter to reduce radius
        # Actually, we want to move them AWAY from the circumcenter to make
        # the cap smaller, or adjust the triangle shape
        # Better: use gradient-free local search on the three vertices
        best_pts = pts.copy()
        best_r = r
        step_size = 0.02

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            # Try small random perturbations of the three vertices
            for _ in range(20):
                trial = pts.copy()
                for vi in [v0, v1, v2]:
                    noise = np.random.randn(3) * step_size
                    trial[vi] = normalize(trial[vi] + noise)
                rad, _ = compute_face_radii(trial)
                if rad is not None:
                    cur_r = rad[face_idx] if face_idx < len(rad) else np.max(rad)
                    if cur_r < best_r:
                        best_r = cur_r
                        best_pts = trial.copy()

            step_size *= 0.95
            if step_size < 0.001:
                break

        return best_pts, best_r < r

    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            points = repulsion_step_density_aware(points, alpha)
            alpha *= 0.999
        return points

    def local_search(points, n_iters=100):
        """Local search targeting the worst facets."""
        best_pts = points.copy()
        best_rad = compute_covering_radius(best_pts)

        for it in range(n_iters):
            if time.time() > deadline - 10.0:
                break

            radii, faces = compute_face_radii(best_pts)
            if radii is None or faces is None:
                break

            # Find worst facet
            worst_idx = np.argmax(radii)
            worst_rad = radii[worst_idx]

            # Try to optimize this facet
            new_pts, improved = optimize_worst_facet(best_pts, worst_idx, faces, max_iter=30)
            if improved:
                best_pts = new_pts
                best_rad = compute_covering_radius(best_pts)

            # Also do a small repulsion step to maintain balance
            best_pts = repulsion_step_density_aware(best_pts, 0.01, radii, faces)

        return best_pts, best_rad

    best_points = None
    best_rad = np.inf

    n_restarts = 4

    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = random_points(N)
        else:
            points = best_points + 0.15 * np.random.randn(N, 3)
            points = normalize(points)

        # Stage 1: Lloyd iteration to balance cell sizes
        if time.time() < deadline - 20.0:
            points = lloyd_iteration(points, n_iter=50)

        # Stage 2: Repulsion-based optimization with density awareness
        alpha = 0.03
        decay = 0.998
        max_iter = 200
        stall_count = 0
        last_rad = np.inf

        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break

            face_radii, faces = compute_face_radii(points)
            points = repulsion_step_density_aware(points, alpha, face_radii, faces)
            alpha *= decay

            if it % 5 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 15:
                        alpha *= 0.9
                        stall_count = 0

            if alpha < 1e-7:
                break

        # Stage 3: Local search on worst facets
        if time.time() < deadline - 10.0 and best_points is not None:
            refined, rad_refined = local_search(best_points.copy(), n_iters=50)
            if rad_refined < best_rad:
                best_rad = rad_refined
                best_points = refined

        # Stage 4: Fine polish
        if time.time() < deadline - 5.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.005, 40)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished

    # Final global polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.003, 60)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        if time.time() < deadline - 3.0 and best_points is not None:
            final2 = polish_step(best_points.copy(), 0.001, 80)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
